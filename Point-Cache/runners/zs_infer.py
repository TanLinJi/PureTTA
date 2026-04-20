import os
import sys
import wandb

import torch

# 将项目根目录加入 Python 搜索路径，保证直接运行脚本时也能导入上层工具函数。
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import *


def infer(args, lm3d_model, test_loader, clip_weights):
    """执行零样本推理，并返回测试集上的平均准确率。

    参数:
        args: 命令行参数集合，主要控制缓存模式、日志频率和 wandb 开关。
        lm3d_model: 已加载的三维编码器模型。
        test_loader: 测试集加载器，每次返回点云、标签和对应的 RGB 特征。
        clip_weights: 文本分类器权重，形状为 (emb_dim, n_cls)。
    """
    # 如果只想评估本地缓存模式，可取消下一行注释做强约束检查。
    # assert args.cache_type == 'local', f'Local cache is expected, but got {args.cache_type}!'

    print('>>> In function `run`: zero-shot inference')

    accuracies = []
    for i, (pc, target, _, rgb) in enumerate(test_loader):
        # pc 和 rgb 都是单个样本的点级输入，形状通常为 (1, n, 3)。
        # 这里把几何坐标与颜色特征在通道维拼接成 (1, n, 6)，作为三维模型的输入。
        feature = torch.cat([pc, rgb], dim=-1).half()

        # 标签搬到 GPU 上，方便后续直接和模型输出计算准确率。
        target = target.cuda()

        # get_logits 会根据 cache_type 返回不同粒度的中间结果：
        # - global: 返回整物体特征 pc_feats
        # - local: 返回局部 patch_centers
        # - hierarchical: 同时返回全局特征和局部中心
        # 其中 clip_logits 是最终用于分类的文本相似度 logits。
        if args.cache_type == 'local':
            patch_centers, clip_logits, loss, prob_map, pred = get_logits(args, feature, lm3d_model, clip_weights)
        elif args.cache_type == 'global':
            pc_feats, clip_logits, loss, prob_map, pred = get_logits(args, feature, lm3d_model, clip_weights)
        elif args.cache_type == 'hierarchical':
            pc_feats, patch_centers, clip_logits, loss, prob_map, pred = get_logits(args, feature, lm3d_model, clip_weights)
        else:
            raise ValueError(f'The choice from [local, global, hierarchical] is expected, but got {args.cache_type}!')

        # 计算当前样本的 top-1 分类准确率，并累积成运行中的平均值。
        acc = cls_acc(clip_logits, target)
        accuracies.append(acc)

        # 逐步向 wandb 记录平均准确率，便于观察推理过程中的波动。
        wandb.log({"Averaged test accuracy": sum(accuracies)/len(accuracies)}, commit=True)
        if i % args.print_freq == 0:
            print("---- Zero-shot test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))

    # 所有测试样本结束后输出最终平均准确率。
    print("---- ***Final*** Zero-shot test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies))) 
    return sum(accuracies)/len(accuracies)


def main(args):
    """主入口：加载模型、准备数据、构建文本分类器并执行零样本测试。"""
    print('>>> In function `main`')

    # 根据 args.lm3d 加载对应的三维模型和 CLIP 文本/图像模型。
    clip_model, lm3d_model = load_models(args)

    # 这里不额外做图像预处理，测试集构建函数内部会根据数据集决定处理方式。
    preprocess = None

    # 当前脚本只处理一个数据集，但仍沿用统一的数据集变量命名。
    dataset_name = args.dataset
    print('>>> In loop `for`')

    print(f"Processing {dataset_name} dataset.")

    # 生成测试集 DataLoader，同时返回类别名和对应的文本模板。
    test_loader, classnames, template = build_test_data_loader(args, dataset_name, args.data_root, preprocess)

    print(f'>>> {[dataset_name]} classnames: {classnames} \n')

    # 将类别名与 prompt template 编码为文本特征，得到 CLIP 分类权重矩阵。
    clip_weights = clip_classifier(args, classnames, template, clip_model)

    # 可选：将本次实验写入 wandb，run_name 会编码模型、数据集和扰动配置。
    if args.wandb:
        if args.lm3d == 'openshape':
            prefix = f"[zs_infer-manual-prompts]/global_feat/{args.lm3d}-{args.oshape_version}"
        elif args.lm3d == 'ulip':
            prefix = f"[zs_infer-manual-prompts]/global_feat/{args.ulip_version}"
        else:
            prefix = f"[zs_infer-manual-prompts]/global_feat/{args.lm3d}"

        # 不同数据集在 run name 中携带的额外信息不同，便于区分实验条件。
        if '_c' in dataset_name and 'sonn' in dataset_name:
            run_name = f"{prefix}/{dataset_name}-{args.sonn_variant}-{args.npoints}/{args.cor_type}"
        elif '_c' in dataset_name:
            run_name = f"{prefix}/{dataset_name}-{args.npoints}/{args.cor_type}"
        elif 'scanobjnn' in dataset_name or 'scanobjectnn' in dataset_name:
            run_name = f"{prefix}/{dataset_name}-{args.sonn_variant}-{args.npoints}"
        elif 'sim2real_sonn' in dataset_name:
            run_name = f"{prefix}/{dataset_name}-{args.sim2real_type}-{args.npoints}"
        elif 'pointda' in dataset_name:
            run_name = f"{prefix}/{dataset_name}-{args.npoints}"
        else:
            run_name = f"{prefix}/{dataset_name}-{args.npoints}"

        run = wandb.init(project="Point-TDA", name=run_name)

    # 执行零样本推理并得到测试集平均准确率。
    zs_acc = infer(args, lm3d_model, test_loader, clip_weights)

    if args.wandb:
        wandb.log({f"{dataset_name}": zs_acc})
        run.finish()


if __name__ == '__main__':
    # 解析命令行参数并固定随机种子，保证测试结果尽可能可复现。
    args = get_arguments()
    set_random_seed(args.seed)

    main(args)

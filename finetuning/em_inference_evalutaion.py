import os
import argparse
from glob import glob
from micro_sam.evaluation.inference import run_instance_segmentation_grid_search
from micro_sam.evaluation.evaluation import run_evaluation


def main():
    parser = argparse.ArgumentParser(description="EM Inference and Evaluation")
    parser.add_argument("-i", "--input_path", required=True, help="数据目录 (含 val/images 和 val/labels)")
    parser.add_argument("-c", "--checkpoint", required=True, help="微调好的模型权重路径 (.pth 或 best.pt)")
    parser.add_argument("-m", "--model_type", default="vit_b", help="模型类型")
    parser.add_argument("-o", "--output_dir", required=True, help="保存预测结果的目录")
    args = parser.parse_args()

    val_image_dir = os.path.join(args.input_path, "val", "images")
    val_label_dir = os.path.join(args.input_path, "val", "labels")

    image_paths = sorted(glob(os.path.join(val_image_dir, "*.tif")))
    gt_paths = sorted(glob(os.path.join(val_label_dir, "*.tif")))

    os.makedirs(args.output_dir, exist_ok=True)

    print("--- 步骤 1: 运行自动网格掩码生成 (Inference) ---")
    # 这会在验证集上进行基于自动网格点提示 (Everything Mode) 的实例分割
    run_instance_segmentation_grid_search(
        predictor=None,  # 将由 checkpoint 和 model_type 自动初始化
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        image_paths=image_paths,
        embedding_dir=os.path.join(args.output_dir, "embeddings"),
        prediction_dir=os.path.join(args.output_dir, "predictions"),
        # 使用针对显微图像优化的网格参数
        grid_search_values={"box_nms_thresh": [0.6, 0.7], "pred_iou_thresh": [0.8, 0.88]}
    )

    print("--- 步骤 2: 计算评估指标 (Evaluation) ---")
    # 计算预测结果与 Ground Truth 之间的指标（默认计算 mAP, mIoU 等）
    pred_paths = sorted(glob(os.path.join(args.output_dir, "predictions", "*.tif")))

    results = run_evaluation(
        gt_paths=gt_paths,
        prediction_paths=pred_paths,
        save_path=os.path.join(args.output_dir, "evaluation_results.csv")
    )

    print("\n评估完成！结果摘要:")
    print(results)
    print(f"详细结果已保存至: {os.path.join(args.output_dir, 'evaluation_results.csv')}")


if __name__ == "__main__":
    main()

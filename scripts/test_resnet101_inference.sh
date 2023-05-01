CUDA_VISIBLE_DEVICES=0 python inference.py \
--vision-backbone resnet101 \
--textual-embeddings embeddings/nih_chest_xray_biobert.npy \
--load-from checkpoints/best_auroc_checkpoint.pth.tar \
--batch-size 4 \
--image-file /Users/sumishra/Downloads/nih_chest_xrays/data/images/00001335_000.png \
--data-root /Users/sumishra/Downloads/nih_chest_xrays

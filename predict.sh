export nnUNet_results="./nnUNetFrame/DATASET/nnUNet_results/"
export nnUNet_raw="./nnUNetFrame/DATASET/nnUNet_results/"
export nnUNet_preprocessed="./nnUNetFrame/DATASET/nnUNet_results/"

python inference_multiprocessing.py -i /workspace/inputs/  -o /workspace/outputs/

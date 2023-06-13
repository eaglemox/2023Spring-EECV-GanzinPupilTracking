python3 inference.py --batch_size 64 \
                     --model_path '.codalab_best.pth' \
                     --data_path './dataset/' --output_path './mask/' --testset True

# for arguments description
# python3 inference.py -h
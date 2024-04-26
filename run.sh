seed_lst=(0 1 2 3 4 5 6 7 8 9)

# small-scale:

# computer: 78.28 ± 0.69
for seed in "${seed_lst[@]}"; do
    python main.py --dataset computer --hetero_train_prop 0.5 --batch_size 1024 --hidden_dim 512 --num_heads 8 \
      --device 1 --num_workers 4 --seed $seed --peak_lr=3e-5  --weight_decay=1e-5 --end_lr=1e-9 --feature_hops 3 --ff_dropout 0.5 --attn_dropout 0.5 \
      --conv_type full --undirected --token_type full
done

# photo: 95.59 ± 0.30
for seed in "${seed_lst[@]}"; do
    python main.py --dataset photo --hetero_train_prop 0.5 --batch_size 1024 --hidden_dim 512 --num_heads 8 \
      --device 1 --num_workers 4 --seed $seed --peak_lr=3e-5  --weight_decay=1e-5 --end_lr=1e-5 --feature_hops 3 --ff_dropout 0.2 --attn_dropout 0.2 \
      --undirected
done

# wikics: 78.28 ± 0.69
for seed in "${seed_lst[@]}"; do
    python main.py --dataset wikics --hetero_train_prop 0.5 --batch_size 1024 --hidden_dim 512 --num_heads 8 \
      --device 2 --num_workers 4 --seed 0 --peak_lr=3e-5  --weight_decay=1e-5 --end_lr=1e-9 --feature_hops 3 --ff_dropout 0.3 --attn_dropout 0.3 \
      --splits_idx $seed --undirected --epoch=2000
done

# roman-empire: 83.71 ± 0.64
for seed in "${seed_lst[@]}"; do
    python main.py --dataset roman-empire --hetero_train_prop 0.5 --batch_size 1024 --hidden_dim 256 --num_heads 8 \
      --device 0 --num_workers 4 --seed 0 --peak_lr=5e-4  --weight_decay=1e-5 --end_lr=1e-9 --feature_hops 3 --ff_dropout 0.5 --attn_dropout 0.5 \
      --splits_idx $seed --conv_type full --token_type full
done

# minesweeper: 90.87 ± 0.44
for seed in "${seed_lst[@]}"; do
    python main.py --dataset minesweeper --hetero_train_prop 0.5 --batch_size 1024 --hidden_dim 128 --num_heads 8 \
      --device 1 --num_workers 4 --seed 0 --peak_lr=1e-4  --weight_decay=0 --end_lr=1e-9  --feature_hops $hop --ff_dropout 0.2 --attn_dropout 0.2 \
      --splits_idx $seed --undirected --max_patience 80 --conv_type full --token_type full
done

# tolokers: 84.07 ± 1.03
for seed in "${seed_lst[@]}"; do
    python main.py --dataset tolokers --hetero_train_prop 0.5 --batch_size 1024 --hidden_dim 64 --num_heads 16 \
      --device 0 --num_workers 4 --seed 0 --peak_lr=1e-3  --weight_decay=0 --end_lr=1e-9 --feature_hops 3 --ff_dropout 0.5 --attn_dropout 0.2 \
      --splits_idx $seed --undirected --conv_type full --token_type full
done

# questions： 77.75 ± 1.26
for seed in "${seed_lst[@]}"; do
    python main.py --dataset questions --hetero_train_prop 0.5 --batch_size 1024 --hidden_dim 512 --num_heads 8 \
      --device 1 --num_workers 4 --seed 0 --peak_lr=2e-4  --weight_decay=1e-5 --end_lr=1e-9 --feature_hops 3 --ff_dropout 0.5 --attn_dropout 0.5 \
      --splits_idx $seed --undirected --conv_type full --token_type full
done


# large-scale:

# ogbn-arxiv: 71.30 ± 0.16
for seed in "${seed_lst[@]}"; do
    python main.py --dataset ogbn-arxiv --hetero_train_prop 0.5 --batch_size 1024 --hidden_dim 256 --num_heads 8 \
      --device 1 --num_workers 4 --seed $seed --peak_lr=8e-4  --weight_decay=1e-5 --end_lr=1e-9 --feature_hops 3 --ff_dropout 0.3 --attn_dropout 0.3 \
      --undirected --warmup_epochs=600 --epoch=1000
done


# pokec: 81.32 ± 0.45
for seed in "${seed_lst[@]}"; do
    python main.py --dataset pokec --hetero_train_prop 0.5 --batch_size 1024 --hidden_dim 128 --num_heads 8 \
      --device 0 --num_workers 4 --seed $seed --peak_lr=5e-3 --weight_decay=1e-5 --end_lr=1e-5 --feature_hops 2 --ff_dropout 0.3 --attn_dropout 0.3 \
      --undirected --warmup_epochs=10 --epoch=500
done


# twitch-gamer: 64.70 ± 0.11
for seed in "${seed_lst[@]}"; do
    python main.py --dataset twitch-gamer --hetero_train_prop 0.5 --batch_size 1024 --hidden_dim 512 --num_heads 8 \
      --device 0 --num_workers 4 --seed $seed --peak_lr=1e-3  --weight_decay=1e-5 --end_lr=1e-9 --feature_hops 3 --ff_dropout 0.3 --attn_dropout 0.2 \
      --undirected
done

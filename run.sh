# munk_A ~ range
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 120 --epochs 150 --seed 2025 -a 0 -n 125
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 120 --epochs 150 --seed 2026 -a 0 -n 125
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 50 --epochs 150 --seed 2025 -a 0 -n 125
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 50 --epochs 150 --seed 2026 -a 0 -n 125
python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 100 --epochs 150 --seed 2025 -a 0 -n 125
python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 100 --epochs 150 --seed 2026 -a 0 -n 125
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 150 --epochs 150 --seed 2025 -a 0 -n 125
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 150 --epochs 150 --seed 2026 -a 0 -n 125
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 200 --epochs 150 --seed 2025 -a 0 -n 125
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 200 --epochs 150 --seed 2026 -a 0 -n 125
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 80 --epochs 150 --seed 2025 -a 0 -n 125
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 80 --epochs 150 --seed 2026 -a 0 -n 125
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 300 --epochs 150 --seed 2025 -a 0 -n 125
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 300 --epochs 150 --seed 2026 -a 0 -n 125
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 30 --epochs 150 --seed 2025 -a 0 -n 125
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 30 --epochs 150 --seed 2026 -a 0 -n 125
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 300 --epochs 150 --seed 2102 -a 0 -n 125
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 200 --epochs 150 --seed 2102 -a 0 -n 125
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 150 --epochs 150 --seed 2102 -a 0 -n 125
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 120 --epochs 150 --seed 2102 -a 0 -n 125
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 100 --epochs 150 --seed 2102 -a 0 -n 125
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 80 --epochs 150 --seed 2102 -a 0 -n 125
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 50 --epochs 150 --seed 2102 -a 0 -n 125
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 30 --epochs 150 --seed 2102 -a 0 -n 125

# 120 km ~ munk_A / munk_B / munk_C
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 120 --epochs 150 --seed 2025 -a 0 -n 125 -s munk_B
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 120 --epochs 150 --seed 2026 -a 0 -n 125 -s munk_B
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 120 --epochs 150 --seed 2025 -a 0 -n 125 -s munk_C
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 120 --epochs 150 --seed 2026 -a 0 -n 125 -s munk_C

# 120 km ~ munk_A -> munk_B / munk_C ~ finetune
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 120 --epochs 50 --seed 2026 -a 0 -n 25 -f --finetune-scenario munk_B --pretrained /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/checkpoints_munk_A_range120km/seed_2026_trainvalnum_125_testnum_500_epoch_150_finetuneFalse_alpha_0.0_schedulercosine/last.pth
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 120 --epochs 50 --seed 2025 -a 0 -n 25 -f --finetune-scenario munk_B --pretrained /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/checkpoints_munk_A_range120km/seed_2025_trainvalnum_125_testnum_500_epoch_150_finetuneFalse_alpha_0.0_schedulercosine/last.pth
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 120 --epochs 50 --seed 2026 -a 0 -n 25 -f --finetune-scenario munk_C --pretrained /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/checkpoints_munk_A_range120km/seed_2026_trainvalnum_125_testnum_500_epoch_150_finetuneFalse_alpha_0.0_schedulercosine/last.pth
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 120 --epochs 50 --seed 2025 -a 0 -n 25 -f --finetune-scenario munk_C --pretrained /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/checkpoints_munk_A_range120km/seed_2025_trainvalnum_125_testnum_500_epoch_150_finetuneFalse_alpha_0.0_schedulercosine/last.pth
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 120 --epochs 50 --seed 2025 -a 0 -n 25 -s munk_B
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 120 --epochs 50 --seed 2026 -a 0 -n 25 -s munk_B
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 120 --epochs 50 --seed 2025 -a 0 -n 25 -s munk_C
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 120 --epochs 50 --seed 2026 -a 0 -n 25 -s munk_C

# 100 km ~ munk_A / munk_B / munk_C
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 100 --epochs 150 --seed 2025 -a 0 -n 125 -s munk_B
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 100 --epochs 150 --seed 2026 -a 0 -n 125 -s munk_B
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 100 --epochs 150 --seed 2102 -a 0 -n 125 -s munk_B
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 100 --epochs 150 --seed 2025 -a 0 -n 125 -s munk_C
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 100 --epochs 150 --seed 2026 -a 0 -n 125 -s munk_C
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 100 --epochs 150 --seed 2102 -a 0 -n 125 -s munk_C

# 100 km ~ munk_A -> munk_B / munk_C ~ finetune
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 100 --epochs 50 --seed 2026 -a 0 -n 25 -f --finetune-scenario munk_B --finetune-range 100 --pretrained /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/checkpoints_munk_A_range100km/seed_2026_trainvalnum_125_testnum_500_epoch_150_finetuneFalse_alpha_0.0_schedulercosine/last.pth
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 100 --epochs 50 --seed 2025 -a 0 -n 25 -f --finetune-scenario munk_B --finetune-range 100 --pretrained /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/checkpoints_munk_A_range100km/seed_2025_trainvalnum_125_testnum_500_epoch_150_finetuneFalse_alpha_0.0_schedulercosine/last.pth
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 100 --epochs 50 --seed 2102 -a 0 -n 25 -f --finetune-scenario munk_B --finetune-range 100 --pretrained /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/checkpoints_munk_A_range100km/seed_2102_trainvalnum_125_testnum_500_epoch_150_finetuneFalse_alpha_0.0_schedulercosine/last.pth
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 100 --epochs 50 --seed 2026 -a 0 -n 25 -f --finetune-scenario munk_C --finetune-range 100 --pretrained /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/checkpoints_munk_A_range100km/seed_2026_trainvalnum_125_testnum_500_epoch_150_finetuneFalse_alpha_0.0_schedulercosine/last.pth
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 100 --epochs 50 --seed 2025 -a 0 -n 25 -f --finetune-scenario munk_C --finetune-range 100 --pretrained /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/checkpoints_munk_A_range100km/seed_2025_trainvalnum_125_testnum_500_epoch_150_finetuneFalse_alpha_0.0_schedulercosine/last.pth
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 100 --epochs 50 --seed 2102 -a 0 -n 25 -f --finetune-scenario munk_C --finetune-range 100 --pretrained /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/checkpoints_munk_A_range100km/seed_2102_trainvalnum_125_testnum_500_epoch_150_finetuneFalse_alpha_0.0_schedulercosine/last.pth
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 100 --epochs 50 --seed 2025 -a 0 -n 25 -s munk_B
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 100 --epochs 50 --seed 2026 -a 0 -n 25 -s munk_B
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 100 --epochs 50 --seed 2102 -a 0 -n 25 -s munk_B
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 100 --epochs 50 --seed 2025 -a 0 -n 25 -s munk_C
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 100 --epochs 50 --seed 2026 -a 0 -n 25 -s munk_C
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 100 --epochs 50 --seed 2102 -a 0 -n 25 -s munk_C

# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 100 --epochs 50 --seed 2025 -a 0 -n 25 -f --finetune-scenario munk_B --finetune-range 100 --pretrained /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/checkpoints_munk_A_range100km/seed_2102_trainvalnum_125_testnum_500_epoch_150_finetuneFalse_alpha_0.0_schedulercosine/last.pth
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 100 --epochs 50 --seed 2026 -a 0 -n 25 -f --finetune-scenario munk_B --finetune-range 100 --pretrained /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/checkpoints_munk_A_range100km/seed_2102_trainvalnum_125_testnum_500_epoch_150_finetuneFalse_alpha_0.0_schedulercosine/last.pth
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 100 --epochs 50 --seed 2025 -a 0 -n 25 -f --finetune-scenario munk_C --finetune-range 100 --pretrained /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/checkpoints_munk_A_range100km/seed_2102_trainvalnum_125_testnum_500_epoch_150_finetuneFalse_alpha_0.0_schedulercosine/last.pth
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 100 --epochs 50 --seed 2026 -a 0 -n 25 -f --finetune-scenario munk_C --finetune-range 100 --pretrained /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/checkpoints_munk_A_range100km/seed_2102_trainvalnum_125_testnum_500_epoch_150_finetuneFalse_alpha_0.0_schedulercosine/last.pth

# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 100 --epochs 150 --seed 2025 -a 0 -n 125
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 100 --epochs 150 --seed 2026 -a 0 -n 125
# python /home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/main.py --range 100 --epochs 150 --seed 2102 -a 0 -n 125

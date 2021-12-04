import os


# data_df = os.path.join(os.path.pardir, 'data', 'metadata', 'db_10_06_21_inc_false_positives.csv')
env_data_dir = os.path.join(os.path.pardir, 'env_data')
mosq_data_dir = os.path.join(os.path.pardir, 'mosq_data')
plot_dir = os.path.join(os.path.pardir, 'outputs',  'plots')
model_dir = os.path.join(os.path.pardir, 'outputs', 'models') # Model sub-directory created in config_keras or config_pytorch
dir_out_MED = os.path.join(os.path.pardir, 'outputs', 'features', 'MED')

rate = 22050
win_size = 30
step_size = 5
n_feat = 128
NFFT = 2048
n_hop = NFFT/4
frame_duration = n_hop/rate # Frame duration in ms
# Normalisation
norm_per_sample = True


min_duration = win_size * frame_duration # Change to match 1.92 (later)

# Create directories if they do not exist:
for directory in [plot_dir, dir_out_MED, model_dir]:
	if not os.path.isdir(directory):
		os.mkdir(directory)
		print('Created directory:', directory)


lpc_order = 16
clip_lpc = True
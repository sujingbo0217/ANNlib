import os
import subprocess
import matplotlib.pyplot as plt

log_dir = 'logs/'
v_recall_stitched, v_qps_stitched = [], []
v_recall_filtered, v_qps_filtered = [], []

print('Missing:')

for filename in os.listdir(log_dir):
    if filename.startswith("stitched") and filename.endswith(".txt"):
        filepath = os.path.join(log_dir, filename)

        recall = subprocess.run(f"grep 'recall:' {filepath} | awk '{{print $2}}'", stdout=subprocess.PIPE, shell=True)
        recall = recall.stdout.decode('utf-8').split('\n')
        recall = [float(v.strip()) for v in recall if v]

        qps = subprocess.run(f"grep 'Find neighbors:' {filepath} | awk '{{print $5}}'", stdout=subprocess.PIPE, shell=True)
        qps = qps.stdout.decode('utf-8').split('\n')
        qps = [float(v.strip()) for v in qps if v]

        # assert(len(recall) == len(qps) == 5)
        if len(recall) != 5 or len(qps) != 5:
            print(filename)
            continue

        v_recall_stitched.append(recall)
        v_qps_stitched.append(qps)
    elif filename.startswith("filtered") and filename.endswith(".txt"):
        filepath = os.path.join(log_dir, filename)

        recall = subprocess.run(f"grep 'recall:' {filepath} | awk '{{print $2}}'", stdout=subprocess.PIPE, shell=True)
        recall = recall.stdout.decode('utf-8').split('\n')
        recall = [float(v.strip()) for v in recall if v]

        qps = subprocess.run(f"grep 'Find neighbors:' {filepath} | awk '{{print $5}}'", stdout=subprocess.PIPE, shell=True)
        qps = qps.stdout.decode('utf-8').split('\n')
        qps = [float(v.strip()) for v in qps if v]

        # assert(len(recall) == len(qps) == 5)
        if len(recall) != 5 or len(qps) != 5:
            print(filename)
            continue

        v_recall_filtered.append(recall)
        v_qps_filtered.append(qps)

# assert len(v_recall_stitched) == len(v_qps_stitched) == len(os.listdir(log_dir))

fig, axs = plt.subplots(1, 5, figsize=(36, 6), gridspec_kw={'wspace': 0.2, 'hspace': 0.2})

colors = ['blue', 'orange', 'purple', 'red', 'green']

subtitles = [
    "100pc specificity",
    "75pc specificity",
    "50pc specificity",
    "25pc specificity",
    "1pc specificity"
]

for i in range(5):
    x_values = [qps[i] for qps in v_qps_stitched]
    y_values = [recall[i] for recall in v_recall_stitched]
    
    axs[i].scatter(x_values, y_values, color=colors[i])
    axs[i].set_title(subtitles[i])
    axs[i].set_xlim(0, 1000)

axs[2].set_xlabel('kQPS', labelpad=12, fontsize=20)
axs[0].set_ylabel('Recall@10', labelpad=40, fontsize=20)

output_path = "./specificity_stitched.pdf"
plt.savefig(output_path, format='pdf', dpi=600)

fig, axs = plt.subplots(1, 5, figsize=(36, 6), gridspec_kw={'wspace': 0.2, 'hspace': 0.2})

for i in range(5):
    x_values = [qps[i] for qps in v_qps_filtered]
    y_values = [recall[i] for recall in v_recall_filtered]
    
    axs[i].scatter(x_values, y_values, color=colors[i])
    axs[i].set_title(subtitles[i])
    axs[i].set_xlim(0, 1000)

axs[2].set_xlabel('kQPS', labelpad=12, fontsize=20)
axs[0].set_ylabel('Recall@10', labelpad=40, fontsize=20)

output_path = "./specificity_filtered.pdf"
plt.savefig(output_path, format='pdf', dpi=600)

plt.show()

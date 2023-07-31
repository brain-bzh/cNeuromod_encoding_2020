import matplotlib.pyplot as plt
import numpy as np
import os
import json
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# TODO: this needs to be filled for each dataset
# because the score differ for each dataset, we need to define the relation for each
KEY_DATASET_SCORE = {
    'dcase2016_task2-hear2021-full': "test", 'gunshot_triangulation-v1.0-full': "aggregated_scores", 'libricount-v1.0.0-hear2021-full': "aggregated_scores", 'maestro-v3.0.0-5h': "aggregated_scores",'mridangam_stroke-v1.5-full': "aggregated_scores", 'mridangam_tonic-v1.5-full': "aggregated_scores", 'nsynth_pitch-v2.2.3-50h': "test", 'nsynth_pitch-v2.2.3-5h': "test", 'speech_commands-v0.0.2-5h': "test", 'speech_commands-v0.0.2-full': "test", 'tfds_crema_d-1.0.0-full': "aggregated_scores", 'tfds_gtzan-1.0.0-full': "aggregated_scores", 'tfds_gtzan_music_speech-1.0.0-full': "aggregated_scores", 'vocal_imitation-v1.1.3-full': "aggregated_scores", 'vox_lingua_top10-hear2021-full': "aggregated_scores", 'beehive_states_fold0-v2-full': "test", 'beehive_states_fold1-v2-full': "test", 'beijing_opera-v1.0-hear2021-full': "aggregated_scores", 'esc50-v2.0.0-full': "aggregated_scores", 'fsd50k-v1.0-full': "test"}
KEY_DATASET_METRIC = {
    'dcase2016_task2-hear2021-full': ["test_event_onset_200ms_fms"], 'gunshot_triangulation-v1.0-full': ["test_top1_acc_mean", "test_top1_acc_std"], 'libricount-v1.0.0-hear2021-full': ["test_top1_acc_mean", "test_top1_acc_std"], 'maestro-v3.0.0-5h': ["test_event_onset_50ms_fms_mean", "test_event_onset_50ms_fms_std"],'mridangam_stroke-v1.5-full': ["test_top1_acc_mean", "test_top1_acc_std"], 'mridangam_tonic-v1.5-full': ["test_top1_acc_mean", "test_top1_acc_std"], 'nsynth_pitch-v2.2.3-50h': ["test_pitch_acc"], 'nsynth_pitch-v2.2.3-5h': ["test_pitch_acc"], 'speech_commands-v0.0.2-5h': ["test_top1_acc"], 'speech_commands-v0.0.2-full': ["test_top1_acc"], 'tfds_crema_d-1.0.0-full': ["test_top1_acc_mean", "test_top1_acc_std"], 'tfds_gtzan-1.0.0-full': ["test_top1_acc_mean", "test_top1_acc_std"], 'tfds_gtzan_music_speech-1.0.0-full': ["test_top1_acc_mean", "test_top1_acc_std"], 'vocal_imitation-v1.1.3-full': ["test_mAP_mean", "test_mAP_std"], 'vox_lingua_top10-hear2021-full': ["test_top1_acc_mean", "test_top1_acc_std"], 'beehive_states_fold0-v2-full': ["test_aucroc"], 'beehive_states_fold1-v2-full': ["test_aucroc"], 'beijing_opera-v1.0-hear2021-full': ["test_top1_acc_mean", "test_top1_acc_std"], 'esc50-v2.0.0-full': ["test_top1_acc_mean", "test_top1_acc_std"], 'fsd50k-v1.0-full': ["test_mAP"]}


def plot_metric(result_path="embeddings/soundnetbrain_hear/", figure_path="reports/figures",doplot=False):

    models = os.listdir(result_path)
    print(models)
    datasets = ([dataset for dataset in os.listdir(
        os.path.join(result_path, models[0], "soundnetbrain_hear"))])
    metric_results = []
    alldict = []
    for model in models:
        metric_models = []
        curdict = dict()
        curdict['model'] = model
        curdict['subject'] = model[:6]
        if 'conv4' in model:
            curdict['finetune'] = 'conv4'
        elif 'conv5' in model:
            curdict['finetune'] = 'conv5'
        elif 'conv6' in model:
            curdict['finetune'] = 'conv6'
        elif 'conv7' in model:
            curdict['finetune'] = 'conv7'
        else:
            curdict['finetune'] = 'no'
        
        if 'MIST_ROI' in model:
            curdict['atlas'] = 'wholebrain'
        else:
            curdict['atlas'] = 'STG'
        for dataset in datasets:
            result_filepath = os.path.join(
                result_path, model, "soundnetbrain_hear",  dataset, "test.predicted-scores.json")
            try:
                with open(result_filepath) as f:
                    data = json.load(f)
                    metric_names = KEY_DATASET_METRIC[dataset]
                    for metric_name in metric_names:
                        score = data[KEY_DATASET_SCORE[dataset]][metric_name]
                        curdict[f"{dataset}_{metric_name}"] = score
                        metric_models += [score]
                        print(f"{metric_name} for {dataset} and {model}:\t{score}")
            except:
                pass
        metric_results += [metric_models]
        alldict.append(curdict)

    Df = pd.DataFrame(alldict).sort_values(by='subject').sort_values(by='model')
    Df.to_csv(os.path.join(figure_path, "metrics.csv"))
    if doplot:
        fig, ax = plt.subplots(figsize=(40, 15))
        cax = ax.matshow(metric_results)
        fig.colorbar(cax, ax=ax)
        for (i, j), z in np.ndenumerate(metric_results):
            ax.text(j, i, '{:0.4f}'.format(z), ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
        # fill x labels with dataset name and associated metric
        x_labels = [dataset_name + "_" +
                    metric_name for dataset_name in datasets for metric_name in KEY_DATASET_METRIC[dataset_name]]
        ax.xaxis.set_ticks(range(len(x_labels)))
        ax.xaxis.set_ticklabels(x_labels, rotation=60, ha="left")
        #ax.xaxis.set_ticks_position("bottom") #use ha="right"
        ax.yaxis.set_ticklabels([""] + models, va="bottom")
        ax.set_xlabel("task")
        ax.set_ylabel("model")
        ax.set_title(
            f"Hear benchmark test metrics")
        plt.savefig(os.path.join(figure_path, "metrics.png"))
    


if __name__ == "__main__":

    root_dir = os.path.join(os.path.dirname(__file__),"..")
    results_path = os.path.join(root_dir,  "embeddings", "soundnetbrain_hear")
    figure_path = os.path.join(root_dir, "reports", "figures")
    plot_metric(result_path=results_path, figure_path=figure_path)

import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import os
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, confusion_matrix
import math
import pickle as pkl
from itertools import combinations
import csv
import logging
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('evaluation_scenarios.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', "%m-%d %H:%M")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

main_dir="."
out_dir = f'{main_dir}/out'
data_dir = f'{main_dir}/data'
pred_dir = f'{data_dir}/predictions'
aligners=["hisat","star"]
samples=["J26675-L1_S1","J26676-L1_S2","J26677-L1_S3","J26678-L1_S4"]
chromosomes=["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","X","Y"]


# soft filtering of given STAR junctions according to https://support.illumina.com/content/dam/illumina-support/help/Illumina_DRAGEN_Bio_IT_Platform_v3_7_1000000141465/Content/SW/Informatics/Dragen/TPipelineSJ_fDG.htm
def illumina_filter(junc):
	# SJ is a noncanonical motif and is only supported by < 3 unique mappings.
	filtered_junc = junc[~((junc.ann==0) & (junc.nr_uniq_reads < 3))]
	# SJ of length > 50000 and is only supported by < 2 unique mappings.
	filtered_junc = filtered_junc[~(((filtered_junc.end-filtered_junc.start)>50000)&(filtered_junc.nr_uniq_reads < 2))]
	# SJ of length > 100000 and is only supported by < 3 unique mappings.
	filtered_junc = filtered_junc[~(((filtered_junc.end-filtered_junc.start)>100000)&(filtered_junc.nr_uniq_reads < 3))]
	# SJ of length > 200000 and is only supported by < 4 unique mappings.
	filtered_junc = filtered_junc[~(((filtered_junc.end-filtered_junc.start)>200000)&(filtered_junc.nr_uniq_reads < 4))]
	# SJ is a noncanonical motif and the maximum spliced alignment overhang is < 30.
	filtered_junc = filtered_junc[~((filtered_junc.ann==0)&(filtered_junc.max_spliced_align_overhang < 30))]
	# SJ is a canonical motif and the maximum spliced alignment overhang is < 12.
	filtered_junc = filtered_junc[~((filtered_junc.ann==1)&(filtered_junc.max_spliced_align_overhang < 12))]
	return filtered_junc


# read in tool predictions for a given tool and junction_type
# output pred_sjs is dictionary of nr_samples*10 DataFrames each containing predicted SJs with columns 'chr','start','end','strand','pred' of junction_type or a Dataframe with columns 'chr','start','end','strand','pred' if junction_type is negatives or annotated
def get_predictions(tool, junction_type, input_aligner=''):
	if (tool=='jcc') & ((input_aligner=='hisat')|(junction_type=='negatives')):
		logger.error(f"{tool} can't be run on {input_aligner} {junction_type} data.")
		exit()
	pred_sjs = {}
	if junction_type in ['50M','500M']:
		for sample_id in samples:
			iterations = (['_'+str(i) for i in range(10)] if junction_type=='50M' else [''])
			for i in iterations:
				if tool in ['deepsplice','spliceai','jcc']:
					pred_sj = pd.read_csv(f'{pred_dir}/{junction_type}/{sample_id}/{input_aligner}/{tool}_pred{i}.csv', dtype={'chr':'object'})
				else:
					logger.error(f"Tool {tool} not supported.")
				pred_sjs[(tool, sample_id, i[1:])] = pred_sj[pred_sj.chr.isin(chromosomes) & pred_sj.strand.isin([1,2])]
	else:
		if junction_type in ['negatives','annotated']:
			pred_sj = pd.read_csv(f'{pred_dir}/{junction_type}/{tool}_pred.csv', dtype={'chr':'object'})
			pred_sjs = pred_sj[pred_sj.chr.isin(chromosomes) & pred_sj.strand.isin([1,2])].drop_duplicates()
	return pred_sjs


# read in ground truth junctions for a given junction_type
# output true_sjs is dictionary of nr_samples DataFrames each containing true SJs with columns 'chr','start','end','strand' / a Dataframe with columns 'chr','start','end','strand' if junction_type is negatives or annotated
def get_gold_standard(junction_type, gt_aligner=None, gt_confidence=None):
	true_sjs = {}
	if junction_type =='500M':
		if (gt_aligner == None) | (gt_confidence == None):
			logger.error(f'For junction type {junction_type} a ground truth aligner and confidence level have to be specified.')
			exit()
		for sample_id in samples:
			true_sj = pd.read_csv(f'{data_dir}/500M/{sample_id}_{gt_aligner}_{gt_confidence}.sj', sep='\t', header = None, usecols=[0,1,2,3], dtype={'chr':'object'}, names = ['chr','start','end','strand'], low_memory=False)
			true_sj['label'] = 1
			true_sjs[sample_id] = true_sj[true_sj.chr.isin(chromosomes) & true_sj.strand.isin([1,2])]
	else:
		if junction_type == '50M':
			logger.error(f'Junction type 50M is no to be used as ground truth.')
			exit()
		if junction_type == 'negatives':
			true_sj = pd.DataFrame(columns=['chr','start','end','strand','label']) #empty DataFrame cause no true sj
		if junction_type == 'annotated':
			true_sj = pd.read_csv(f'{data_dir}/annotated.sj',sep='\t', header=None, names=['chr','start','end','strand'], dtype={'chr':'object', 'start':int, 'end':int, 'strand':int})
			true_sj['label'] = 1
		true_sjs = true_sj[true_sj.chr.isin(chromosomes) & true_sj.strand.isin([1,2])]
	return true_sjs


# read in input positions for 50M data
def get_50M_input_pos(input_aligner):
	in_sjs = {}
	for sample_id in samples:
		for i in range(10):
			in_sj = pd.read_csv(f'{data_dir}/50M/{input_aligner}/{sample_id}_50M_{i}.sj',sep='\t',header = None, usecols= [0,1,2,3], dtype={0:'object'}, names = ['chr','start','end','strand'],low_memory=False) #just take all annotated SJs as true, we don't use tool prediction
			in_sjs[(sample_id, i)] = in_sj[in_sj.chr.isin(chromosomes) & in_sj.strand.isin([1,2])]
	return in_sjs


# round std up and keep 2 significant digits
# round mean to same precision as std. dev., if more than 3 digits after comma use e notation
def log_round_mean_std(mean, std, tool, aligner, gt_confidence, score):
	nr_nks_std = int(f"{std:.1e}".split('e')[1][1:]) +  1
	std += round(0.1**(nr_nks_std))
	std = round(std, nr_nks_std)
	mean = round(mean, nr_nks_std)
	if std <= 0.009:
		logger.debug(f"{tool} {aligner} {gt_confidence} {score}: {mean} ± {std:.1E}")
	else:
		logger.debug(f"{tool} {aligner} {gt_confidence} {score}: {mean} ± {std}")


# save mean No Skill AUPRC +- 1 std. dev. (over all tools for scenario) in file
def calc_no_skill_auprc(tool_sjs, score_dir, aligner, gt_confidence):
	perc_positives = []
	for tool, (_, sjs) in enumerate(tool_sjs.items()):
		for _, sj in sjs.items():
			perc_positives.append(len(sj[sj['label']==1])/len(sj))
	# Since precision is constant for a No-Skill Classifier, AUPRC is the constant precision multiplied by the range of recall (which is 1)
	no_skill_auprc = np.mean(perc_positives)
	no_skill_std = np.std(perc_positives)
	log_round_mean_std(no_skill_auprc, no_skill_std, 'No Skill', aligner, gt_confidence, 'AUPRC')
	out_file = f'{score_dir}/No_Skill_auprcs.csv'
	if (not os.path.isfile(out_file)):
		with open(out_file, 'w') as file:
			file.write(f'aligner,gt_confidence,tool,mean_auprc,std_auprc\n')
	with open(out_file, 'a') as f:
		f.write(f"{aligner},{gt_confidence},No Skill,{no_skill_auprc},{no_skill_std}\n")


def calc_f1_score_at_threshold(sjs, score_dir, aligner, gt_confidence, threshold=0.5):
	threshold = round(threshold, 1) # round threshold to 1 decimal place
	threshold_ = threshold 
	f1_scores = {}
	for ann, sj in sjs.items():
		tool, _,_ = ann
		# f1 score at given threshold
		if tool == 'deepsplice':
			threshold_ = threshold * 1.7052 # scale DeepSplice scores to [0,1] range
		f1_scores[ann] = f1_score(sj['label'], sj['pred'] >= threshold_)
	mean_f1_score = np.mean(list(f1_scores.values()))
	std_f1_score = np.std(list(f1_scores.values()))
	# if out_mean_file does not exist, create it and write header
	out_mean_file = f'{score_dir}/mean_std_f1_scores.csv'
	if (not os.path.isfile(out_mean_file)):
		with open(out_mean_file, 'w') as file:
			file.write(f'aligner,gt_confidence,tool,mean_f1_score,std_f1_score,threshold\n')
	with open(out_mean_file, 'a') as file:
		file.write(f'{aligner},{gt_confidence},{tool},{mean_f1_score},{std_f1_score},{threshold}\n')
	log_round_mean_std(mean_f1_score, std_f1_score, tool, aligner, gt_confidence, f'F1 Score at {threshold}')


# save mean AUPRC (area under the precision recall curve) scores +- 1 std. dev. and number positives and number negatives +- 1 std. dev. in file 
def calc_auprc(sjs, score_dir, aligner, gt_confidence):
	scores = {}
	nr_positives = []
	nr_negatives = []
	for ann, sj in sjs.items():
		tool, _,_ = ann
		score = average_precision_score(sj['label'], sj['pred'])
		scores[ann] = score
		nr_positives.append(len(sj[sj["label"]==1]))
		nr_negatives.append(len(sj[sj["label"]==0]))
	mean_auprc = np.mean(list(scores.values()))
	auprc_std = np.std(list(scores.values()))
	# if out_file does not exist, create it and write header
	out_file = f'{score_dir}/{tool}_{aligner}_{gt_confidence}_auprcs.csv'
	with open(out_file, 'w') as file:
		file.write(f'tool,sample_id,run_nr,auprc\n')
	with open(out_file, 'a') as file:
		for ann, score in scores.items():
			file.write(','.join(ann)+','+str(score)+'\n')
	# if out_mean_file does not exist, create it and write header
	out_mean_file = f'{score_dir}/mean_std_auprcs.csv'
	if (not os.path.isfile(out_mean_file)):
		with open(out_mean_file, 'w') as file:
			file.write(f'aligner,gt_confidence,tool,mean_auprc,std_auprc\n')
	with open(out_mean_file, 'a') as file:
		file.write(f'{aligner},{gt_confidence},{tool},{mean_auprc},{auprc_std}\n')
	log_round_mean_std(mean_auprc, auprc_std, tool, aligner, gt_confidence, 'AUPRC')
	# if out_nr_file does not exist, create it and write header
	out_nr_file = f'{score_dir}/mean_nr_positives_negatives.csv'
	if (not os.path.isfile(out_nr_file)):
		with open(out_nr_file, 'w') as file:
			file.write(f'aligner,gt_confidence,tool,mean_nr_positives,std_nr_positives,mean_nr_negatives,std_nr_negatives\n')
	with open(out_nr_file, 'a') as file:
		file.write(f'{aligner},{gt_confidence},{tool},{math.floor(np.mean(nr_positives))},{math.floor(np.std(nr_positives))},{math.floor(np.mean(nr_negatives))},{math.floor(np.std(nr_negatives))}\n')


# save effectsize in file 
def write_effectsize_ci(tool_sjs, stats_dir, aligner, gt_confidence, n_bootstrap=1000):
    TOOL_PLOT_NAME = {
        'spliceai': 'SpliceAI',
        'deepsplice': 'DeepSplice',
        'jcc': 'JCC',
        'baseline': 'No-Skill'
    }
    # Tools to use (excluding baseline)
    tool_names = [key for key in TOOL_PLOT_NAME if key in tool_sjs]

    # Precompute bootstrap indices for every sample/run
    bootstrap_indices = {}
    lens = {}
    for (_, sample, run_id), s in tool_sjs[tool_names[0]].items():
        n = s.shape[0]
        bootstrap_indices[(sample, run_id)] = np.random.randint(0, n, size=(n_bootstrap, n))
        lens[(sample, run_id)] = n

    # Precompute per-tool per-(sample,run) AUPRCs for each bootstrap
    per_tool_auprcs = {t: {} for t in tool_names}
    for t in tool_names:
        for (_, sample, run_id), s in tool_sjs[t].items():
            indices = bootstrap_indices[(sample, run_id)]
            labels = s['label'].values
            preds = s['pred'].values
            boot_auprcs = np.array([
                average_precision_score(labels[idx], preds[idx]) for idx in indices
            ])
            per_tool_auprcs[t][(sample, run_id)] = boot_auprcs

    # Precompute baseline auprcs:
    per_baseline_auprcs = {}
    for (_, sample, run_id), s in tool_sjs[tool_names[0]].items():
        indices = bootstrap_indices[(sample, run_id)]
        labels = s['label'].values
        n = len(labels)
        # Baseline: predict all 1's
        preds_baseline = np.ones_like(labels)
        boot_auprcs = np.array([
            average_precision_score(labels[idx], preds_baseline[idx]) for idx in indices
        ])
        per_baseline_auprcs[(sample, run_id)] = boot_auprcs

    results_rows = []

    # Compare each tool to baseline
    for t in tool_names:
        t_name = TOOL_PLOT_NAME[t]
        for (sample, run_id) in per_tool_auprcs[t]:
            tool_auprcs = per_tool_auprcs[t][(sample, run_id)]
            baseline_auprcs = per_baseline_auprcs[(sample, run_id)]
            bootstrapped_deltas = tool_auprcs - baseline_auprcs
            mean_delta = np.mean(bootstrapped_deltas)
            ci_lower, ci_upper = np.percentile(bootstrapped_deltas, [2.5, 97.5])
            row = [aligner, gt_confidence, sample, run_id, t_name, TOOL_PLOT_NAME['baseline'], mean_delta, ci_lower, ci_upper]
            results_rows.append(row)

    # Compare all pairs of tools (excluding baseline)
    for t1, t2 in combinations(tool_names, 2):
        n1 = TOOL_PLOT_NAME[t1]
        n2 = TOOL_PLOT_NAME[t2]
        for (sample, run_id) in per_tool_auprcs[t1]:
            auprc1 = per_tool_auprcs[t1][(sample, run_id)]
            auprc2 = per_tool_auprcs[t2][(sample, run_id)]
            bootstrapped_deltas = auprc1 - auprc2
            mean_delta = np.mean(bootstrapped_deltas)
            ci_lower, ci_upper = np.percentile(bootstrapped_deltas, [2.5, 97.5])
            row = [aligner, gt_confidence, sample, run_id, n1, n2, mean_delta, ci_lower, ci_upper]
            results_rows.append(row)

    # Write to CSV
    csv_file_path = f'{stats_dir}/effect_sizes.csv'
    write_header = not os.path.exists(csv_file_path) or os.stat(csv_file_path).st_size == 0
    with open(csv_file_path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            headers = ['aligner', 'gt_confidence', 'sample', 'run_id', 'tool1', 'tool2', 'mean_delta', 'ci_lower', 'ci_upper']
            writer.writerow(headers)
        writer.writerows(results_rows)


def write_operating_points(y_true, y_scores, stats_dir, tool, aligner, gt_confidence):
	# At fixed thresholds
	thresholds = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
	summary_file = f"{stats_dir}/operating_points.csv"
	if (not os.path.isfile(summary_file)):
		with open(summary_file, 'w') as file:
			file.write(f'tool,aligner,gt_confidence,threshold,precision,recall,TP,FP,FN,TN\n')
	with open(summary_file, 'a') as f:
		for threshold in thresholds:
			pred_at_thresh = (y_scores >= threshold).astype(int)
			precision_fixed = precision_score(y_true, pred_at_thresh, zero_division=0)
			recall_fixed = recall_score(y_true, pred_at_thresh, zero_division=0)
			tn, fp, fn, tp = confusion_matrix(y_true, pred_at_thresh).ravel()
			f.write(f"{tool},{aligner},{gt_confidence},{threshold},{precision_fixed},{recall_fixed},{tp},{fp},{fn},{tn}\n")


# calculate performance for Scenario 1b: "Detecting spurious junctions"
def scenario_1b(aligner, gt_confidence, stats_dir, file_sj_50):
	os.makedirs(stats_dir, exist_ok=True)
	if os.path.isfile(file_sj_50):
		with open(file_sj_50, 'rb') as f:
			tool_sjs_50 = pkl.load(f)
		logger.debug(f'Loaded {file_sj_50} from disk.')
	else:
		true_sjs_500 = get_gold_standard('500M', aligner, gt_confidence)
		tool_sjs_50 = {}
		input_pos_50 = get_50M_input_pos(aligner)
		tools = ['deepsplice','spliceai','jcc']
		if aligner == 'hisat': tools = [t for t in tools if t != 'jcc'] # tool JCC can't be run on HISAT aligned data
		for tool in tools:
			pred_sjs_50 = get_predictions(tool, '50M', aligner)
			sjs_50 = {}
			for pred_ann,pred_sj in pred_sjs_50.items():
				_ , sample_id, run_id = pred_ann
				run_id = int(run_id)
				input_pos = input_pos_50[(sample_id,run_id)]
				true_sj = true_sjs_500[sample_id]
				# take all 50M positions 
				merged_sj = input_pos.merge(true_sj, on=['chr', 'start', 'end', 'strand'], how='left')
				# left merge, treat those positions as prediction 0
				merged_sj = merged_sj.merge(pred_sj, on=['chr', 'start', 'end', 'strand'], how='left')
				# if tool doesn't give prediction for positions? -> use pred 0
				if len(merged_sj[merged_sj['pred'].isna()]) != 0:
					logger.debug(f"{pred_ann}: {len(merged_sj[merged_sj['pred'].isna()])}/{len(merged_sj)} ({((len(merged_sj[merged_sj['pred'].isna()])/len(merged_sj))*100):.2f} %) {aligner} junctions don't have predictions.")
					merged_sj['pred'] = merged_sj['pred'].fillna(0)
				# if tool gives prediction for positions that we dont have labels for -> treat those positions as label 0
				if len(merged_sj[merged_sj['label'].isna()]) != 0:
					logger.debug(f"{pred_ann}: {len(merged_sj[merged_sj['label'].isna()])}/{len(merged_sj)} ({((len(merged_sj[merged_sj['label'].isna()])/len(merged_sj))*100):.2f} %) predicted junctions are not in the GT.")
					merged_sj['label'] = merged_sj['label'].fillna(0).astype(int)
				# Flip prediction scores
				merged_sj['pred'] = 1 - merged_sj['pred']
				# Flip labels
				merged_sj['label'] = 1 - merged_sj['label']
				sjs_50[pred_ann] = merged_sj
			tool_sjs_50[tool] = sjs_50
		with open(file_sj_50, 'wb') as f:
			pkl.dump(tool_sjs_50, f)
	for tool, sjs in tool_sjs_50.items():
		all_labels = np.concatenate([sj["label"] for sj in sjs.values()]) # append all samples
		all_preds = np.concatenate([sj["pred"] for sj in sjs.values()]) # append all samples
		write_operating_points(all_labels, all_preds, stats_dir, tool, aligner, gt_confidence)
		calc_auprc(sjs, stats_dir, aligner, gt_confidence)
		for threshold in np.arange(0, 1.1, 0.1):
			calc_f1_score_at_threshold(sjs, stats_dir, aligner, gt_confidence, threshold=threshold)
	calc_no_skill_auprc(tool_sjs_50, stats_dir, aligner, gt_confidence)
	write_effectsize_ci(tool_sjs_50, stats_dir, aligner, gt_confidence)
	return tool_sjs_50


# calculate performance for Scenario 1a: "Detecting junctions from a decoy dataset"
def scenario_1a(aligner, gt_confidence, stats_dir, file_sj_50_neg, tool_sjs_50):
	os.makedirs(stats_dir, exist_ok=True)
	if os.path.isfile(file_sj_50_neg):
		with open(file_sj_50_neg, 'rb') as f:
			tool_sjs_50_neg = pkl.load(f)
		logger.debug(f'Loaded {file_sj_50_neg} from disk.')
	else:
		true_sjs_neg = get_gold_standard('negatives') # empty DataFrame
		tool_sjs_50_neg = {}
		for tool in ['deepsplice','spliceai']:
			pred_sjs_neg = get_predictions(tool, 'negatives')
			sjs_neg = pred_sjs_neg.merge(true_sjs_neg, on=['chr', 'start', 'end', 'strand'], how='outer') #get label + pred for negatives
			# if tool gives prediction for positions that we dont have labels for -> treat those positions as label 0
			sjs_neg['label'] = sjs_neg['label'].fillna(0).astype(int)
			sjs_50_neg = {}
			for ann_50, sj_50 in tool_sjs_50[tool].items():
				nr_positives = len(sj_50[sj_50.label == 1]) # positives (technical noise) 
				negatives = sj_50[sj_50.label == 0] # negatives (according to filtered GT)
				sjs_neg_sub = sjs_neg.sample(n=nr_positives, random_state=42) # subset generated negatives to same nr sj
				# Flip prediction scores
				sjs_neg_sub['pred'] = 1 - sjs_neg_sub['pred']
				# Flip labels
				sjs_neg_sub['label'] = 1 - sjs_neg_sub['label']
				sj_sets = [sjs_neg_sub, negatives]
				merged_sj = pd.concat(sj_sets,ignore_index=True)
				if len(merged_sj[merged_sj['label'].isna()]) != 0:
					logger.debug(f"{ann_50}: {len(merged_sj[merged_sj['label'].isna()])}/{len(merged_sj)} ({((len(merged_sj[merged_sj['label'].isna()])/len(sjs_neg))*100):.2f} %) predicted junctions are not in the GT.")
					merged_sj['label'] = merged_sj['label'].fillna(0).astype(int)
				if len(merged_sj[merged_sj['pred'].isna()]) != 0:
					logger.debug(f"{ann_50}: {len(merged_sj[merged_sj['pred'].isna()])}/{len(merged_sj)} ({((len(merged_sj[merged_sj['pred'].isna()])/len(merged_sj))*100):.2f} %) {aligner} junctions don't have predictions.")
					merged_sj['pred'] = merged_sj['pred'].fillna(0)
				sjs_50_neg[ann_50] = merged_sj
			tool_sjs_50_neg[tool] = sjs_50_neg
		with open(file_sj_50_neg, 'wb') as f:
			pkl.dump(tool_sjs_50_neg, f)
	for tool, sjs in tool_sjs_50_neg.items():
		all_labels = np.concatenate([sj["label"] for sj in sjs.values()]) # append all samples
		all_preds = np.concatenate([sj["pred"] for sj in sjs.values()]) # append all samples
		write_operating_points(all_labels, all_preds, stats_dir, tool, aligner, gt_confidence)
		calc_auprc(sjs, stats_dir, aligner, gt_confidence)
		for threshold in np.arange(0, 1.1, 0.1):
			calc_f1_score_at_threshold(sjs, stats_dir, aligner, gt_confidence, threshold=threshold)
	calc_no_skill_auprc(tool_sjs_50_neg, stats_dir, aligner, gt_confidence)
	write_effectsize_ci(tool_sjs_50_neg, stats_dir, aligner, gt_confidence)


# calculate performance for Scenario 2: "Predicting junctions that could be detected with higher sequencing depth" / Scenario 3: "Predicting hard-to-find junctions"
# for Real-world / Hypothetical setting
def scenario_2_3(aligner, gt_confidence, stats_dir, scenario, hard_to_find, file_sj_50):
	os.makedirs(stats_dir, exist_ok=True)
	if os.path.isfile(file_sj_50):
		with open(file_sj_50, 'rb') as f:
			tool_sjs_50 = pkl.load(f)
		logger.debug(f'Loaded {file_sj_50} from disk.')
	else:
		true_sjs_500 = get_gold_standard('500M', aligner, gt_confidence)
		input_pos_50 = get_50M_input_pos(aligner)
		input_pos_50_sample = {}
		for sample_id in samples:
			# for each sample combine all input positions that are in any of the 10 subsampled 50M runs
			input_pos_50_sample[sample_id] = pd.concat([input_pos_50[(sample_id,run_id)] for run_id in range(10)]).drop_duplicates()
		label_annotated = get_gold_standard('annotated')
		tool_sjs_50 = {}
		tools = ['deepsplice','spliceai'] if (scenario == 'hypothetical') else ['jcc']
		for tool in (tools if aligner != 'hisat' else [t for t in tools if t != 'jcc']): # JCC can't be run on HISAT aligned data
			if scenario == 'hypothetical':
				pred_sjs_50 = get_predictions(tool, '500M', aligner)
			elif scenario == 'real_world':
				pred_sjs_50 = get_predictions(tool, '50M', aligner)
			sjs_50 = {}
			for pred_ann,pred_sj in pred_sjs_50.items():
				_, sample_id, run_id = pred_ann
				# get all input positions of any of the 10 subsampled 50M runs
				input_pos = input_pos_50_sample[sample_id]
				true_sj = true_sjs_500[sample_id]
				merged_sj = pred_sj.merge(true_sj, on=['chr', 'start', 'end', 'strand'], how='outer' if scenario=='real_world' else 'right')
				if hard_to_find:
					# - positions given in 50M - positions annotated in reference genome
					merged_sj = merged_sj.merge(input_pos, on=['chr', 'start', 'end', 'strand'], how='left', indicator=True)
					merged_sj = merged_sj[merged_sj._merge == 'left_only'].drop(columns='_merge')
					merged_sj = merged_sj.merge(label_annotated[['chr', 'start', 'end', 'strand']], on=['chr', 'start', 'end', 'strand'], how='left', indicator=True)
					merged_sj = merged_sj[merged_sj._merge == 'left_only'].drop(columns='_merge')
				if len(merged_sj[merged_sj['pred'].isna()]) != 0:
					logger.debug(f"{pred_ann}: {len(merged_sj[merged_sj['pred'].isna()])}/{len(merged_sj)} ({((len(merged_sj[merged_sj['pred'].isna()])/len(merged_sj))*100):.2f} %) {aligner} junctions don't have predictions.")
					merged_sj['pred'] = merged_sj['pred'].fillna(0)
				if len(merged_sj[merged_sj['label'].isna()]) != 0:
					logger.debug(f"{pred_ann}: {len(merged_sj[merged_sj['label'].isna()])}/{len(merged_sj)} ({((len(merged_sj[merged_sj['label'].isna()])/len(merged_sj))*100):.2f} %) predicted junctions are not in the GT.")
					merged_sj['label'] = merged_sj['label'].fillna(0).astype(int)
				sjs_50[pred_ann] = merged_sj # pred_ann [(tool, sample_id, i)]
			tool_sjs_50[tool] = sjs_50
		if scenario == 'hypothetical':
			tool_sjs_50 = scenario_2_3_add_negatives(tools, tool_sjs_50)
		with open(file_sj_50, 'wb') as f:
			pkl.dump(tool_sjs_50, f)
	for tool, sjs in tool_sjs_50.items():
		all_labels = np.concatenate([sj["label"] for sj in sjs.values()]) # append all samples
		all_preds = np.concatenate([sj["pred"] for sj in sjs.values()]) # append all samples
		write_operating_points(all_labels, all_preds, stats_dir, tool, aligner, gt_confidence)
		calc_auprc(sjs, stats_dir, aligner, gt_confidence)
		for threshold in np.arange(0, 1.1, 0.1):
			calc_f1_score_at_threshold(sjs, stats_dir, aligner, gt_confidence, threshold=threshold)
	calc_no_skill_auprc(tool_sjs_50, stats_dir, aligner, gt_confidence)
	write_effectsize_ci(tool_sjs_50, stats_dir, aligner, gt_confidence)


# for Hypothetical setting we add negative junctions for evaluation
def scenario_2_3_add_negatives(tools, tool_sjs_50):
	# add same amount negatives 
	true_sjs_neg = get_gold_standard('negatives') # empty DataFrame
	tool_sjs_50_neg = {}
	for tool in tools:
		pred_sjs_neg = get_predictions(tool, 'negatives')
		sjs_neg = pred_sjs_neg.merge(true_sjs_neg, on=['chr', 'start', 'end', 'strand'], how='outer') # predictions negatives + true negatives
		sjs_neg['label'] = sjs_neg['label'].fillna(0).astype(int)
		if len(sjs_neg[sjs_neg['pred'].isna()]) != 0:
			logger.error(f'Error: {tool} some of the predictions are NaN.')
			exit()
		sjs_50_neg = {}
		for ann_50, sj_50 in tool_sjs_50[tool].items():
			if len(sjs_neg) < len(sj_50):
				logger.debug(f"{ann_50}: We can't sample {len(sj_50)} positions. We take all {len(sjs_neg)} negative positions instead.")
			else:
				logger.debug(f"{ann_50}: sample {len(sj_50)} negative positions.") 
			sjs_neg_sub = sjs_neg.sample(n=min(len(sj_50),len(sjs_neg)), random_state=42) # subset negatives to same nr sj
			merged_sj = pd.concat([sj_50, sjs_neg_sub],ignore_index=True) # concat with data from before
			sjs_50_neg[ann_50] = merged_sj
		tool_sjs_50_neg[tool] = sjs_50_neg
	return tool_sjs_50_neg


# calculate AUPRC + number positives + number negatives for each scenario
stats_dir = f'{out_dir}/stats'
os.makedirs(stats_dir, exist_ok = True)

for aligner, gt_confidence in [('star','unfiltered'),('star','illumina'),('star','cutoff'),('hisat','unfiltered')]:

	logger.info(f'------------ RUN Scenario 1b vs GT {aligner} {gt_confidence} ------------')
	stats_dir_a = f'{stats_dir}/scenario_1b'
	file_sj_50 = f'{stats_dir_a}/tool_sjs_50_{aligner}_vs_GT_{aligner}_{gt_confidence}.pkl'
	tool_sjs_50 = scenario_1b(aligner, gt_confidence, stats_dir_a, file_sj_50)

	logger.info(f'------------ RUN Scenario 1a vs GT {aligner} {gt_confidence} ------------')
	stats_dir_b = f'{stats_dir}/scenario_1a'
	file_sj_50_neg_base = f'{stats_dir_b}/tool_sjs_50_{aligner}_vs_GT_{aligner}_{gt_confidence}.pkl'
	scenario_1a(aligner, gt_confidence, stats_dir_b, file_sj_50_neg_base, tool_sjs_50)

for hard_to_find in [False, True]:
	for scenario in ['hypothetical','real_world']:
		for aligner, gt_confidence in [('star','unfiltered'),('star','illumina'),('star','cutoff'),('hisat','unfiltered')]:
			if not ((scenario == 'real_world') and (aligner == 'hisat')):
				logger.info(f'--------- RUN Scenario {3 if hard_to_find else 2} {scenario} vs GT {aligner} {gt_confidence} ---------')
				stats_dir_scenario = f'{stats_dir}/scenario_{3 if hard_to_find else 2}_{scenario}'
				os.makedirs(stats_dir_scenario, exist_ok = True)
				file_sj_50 = f'{stats_dir_scenario}/tool_sjs_50_{aligner}_vs_GT_{aligner}_{gt_confidence}.pkl' 
				scenario_2_3(aligner, gt_confidence, stats_dir_scenario, scenario, hard_to_find, file_sj_50)

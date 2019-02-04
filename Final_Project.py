import numpy as np
import scipy.sparse as sp
import scipy.io as spio
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import statistics as stats
import math as math
import operator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from scipy import integrate

def mutation_data_parser(original_file, new_file):
    with open(original_file,"r") as stuff_to_write:
        with open(new_file,"w") as stuff_written: 
            for line in stuff_to_write:
                if "ss_pick" not in line and "prot_acc=." not in line and "SEQ" not in line:
                    stuff_written.write(line)
    sample_snp_id = []
    sample_gene_abbrev = []
    sample_fxn_class = []
    sample_residue = []
    sample_aa_position = []
    sample_protein_acc = []
    with open(new_file,"r") as sample_info:
        for line in sample_info:
            if "rs" and "Homo sapiens" in line:
                sample_snp_id.append(line.split(" | ")[0])
            elif "LOC |"and "residue" in line and "fxn-class=reference" not in line:
                sample_gene_abbrev.append(line.split(" | ")[1])
                sample_fxn_class.append((line.split(" | ")[3])[10:])
                sample_residue.append((line.split(" | ")[6])[8:])
                sample_aa_position.append((line.split(" | ")[7])[12:])
                sample_protein_acc.append((line.split(" | ")[-1])[9:-1])
    return(sample_snp_id, sample_gene_abbrev, sample_fxn_class, sample_residue, sample_aa_position, sample_protein_acc)

sample_benign_snp_id, sample_benign_gene_abbrev, sample_benign_fxn_class, sample_benign_residue, sample_benign_aa_position, sample_benign_protein_acc = mutation_data_parser("./sample_benign.txt", "./sample_benign_new.txt")
sample_pathogenic_snp_id, sample_pathogenic_gene_abbrev, sample_pathogenic_fxn_class, sample_pathogenic_residue, sample_pathogenic_aa_position, sample_pathogenic_protein_acc = mutation_data_parser("./sample_pathogenic.txt", "./sample_pathogenic_new.txt")
benign_snp_id, benign_gene_abbrev, benign_fxn_class, benign_residue, benign_aa_position, benign_protein_acc = mutation_data_parser("./Benign_nsSNP_Protein_Available.txt", "./New_Benign_nsSNP_Protein_Available.txt")
pathogenic_snp_id, pathogenic_gene_abbrev, pathogenic_fxn_class, pathogenic_residue, pathogenic_aa_position, pathogenic_protein_acc = mutation_data_parser("./Pathogenic_nsSNP_Protein_Available.txt", "./New_Pathogenic_nsSNP_Protein_Available.txt")
    
def appending_snp_id(snp_id, gene_abbrev):
    new_snp_id = []
    snp_id_select = 0
    new_snp_id.append(snp_id[0])
    for i_iter in range(1, len(gene_abbrev)):
        if gene_abbrev[i_iter] == gene_abbrev[i_iter - 1]:
            new_snp_id.append(snp_id[snp_id_select])
        elif gene_abbrev[i_iter] != gene_abbrev[i_iter - 1]:
            snp_id_select += 1
            new_snp_id.append(snp_id[snp_id_select])
    return(new_snp_id)

new_sample_benign_id = appending_snp_id(sample_benign_snp_id, sample_benign_gene_abbrev)
new_sample_pathogenic_id = appending_snp_id(sample_pathogenic_snp_id, sample_pathogenic_gene_abbrev)
new_pathogenic_id = appending_snp_id(pathogenic_snp_id, pathogenic_gene_abbrev)
new_benign_id = appending_snp_id(benign_snp_id, benign_gene_abbrev)

amino_acids = ["G","A","V","L","I","P","F","Y","W","S","T","N","Q","C","M","D","E","H","K","R"]
hydrophobic_values = [0.67, 1.0, 2.3, 2.2, 3.1, -0.29, 2.5, 0.08, 1.5, -1.1, -0.75, -2.7, -2.9, 0.17, 1.1, -3.0, -2.6, -1.7, -4.6, -7.5]
molec_mass_values = [57, 71, 99, 113, 113, 97, 147, 163, 186, 87, 101, 114, 128, 103, 131, 115, 129, 137, 128, 156]
charge_values = [0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0.5, 0, -1, -1, 0.5, 1, 1]
pI_values = [5.97, 6.01, 5.97, 5.98, 6.02, 6.48, 5.48, 5.66, 5.89, 5.68, 5.87, 5.41, 5.65, 5.07, 5.74, 2.77, 3.22, 7.59, 9.74, 10.76]
proline_present = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
hydro_dict = dict(zip(amino_acids, hydrophobic_values)) 
mm_dict = dict(zip(amino_acids, molec_mass_values))
cv_dict = dict(zip(amino_acids, charge_values))
pi_dict = dict(zip(amino_acids, pI_values))
pro_dict = dict(zip(amino_acids, proline_present))

def parse_all_proteins():
    sequences = {}
    in_sequence = False
    current_sequence_name = ""
    with open("./All_Proteins.fasta","r") as sample_fasta:
        for line in sample_fasta.readlines():
            if line[0] == '>':
                in_sequence = True
                current_sequence_name = line.split(" ")[0].split(">")[1]
                sequences[current_sequence_name] = []
            elif line == "\n":
                sequences[current_sequence_name] = "".join(sequences[current_sequence_name])
                in_sequence = False
                current_sequence_name = ""
            elif in_sequence is True:
                sequences[current_sequence_name].append(line.strip())
    return(sequences)

sequence_dict = parse_all_proteins()

def create_dataframe(sample_snp_id, sample_protein_acc, sample_gene_abbrev, sample_fxn_class, sample_residue, sample_aa_position, sequence_dict, hydro_dict, mm_dict, cv_dict, pro_dict):
    sample_dataframe = pd.DataFrame(np.column_stack([sample_snp_id, sample_protein_acc, sample_gene_abbrev, sample_fxn_class, sample_residue, sample_aa_position]),
                                     columns = ["snp_id", "prot_acc", "gene_abb", "class", "residue", "position"])
    sample_dataframe["prim_seq"] = sample_dataframe["prot_acc"].map(sequence_dict)
    sample_dataframe["hydro_vals"] = sample_dataframe["residue"].map(hydro_dict)
    sample_dataframe["mw_vals"] = sample_dataframe["residue"].map(mm_dict)
    sample_dataframe["charge_vals"] = sample_dataframe["residue"].map(cv_dict)
    sample_dataframe["pi_vals"] = sample_dataframe["residue"].map(pi_dict)
    sample_dataframe["pro_pres"] = sample_dataframe["residue"].map(pro_dict)
    return(sample_dataframe)

sample_benign_dataframe = create_dataframe(new_sample_benign_id, sample_benign_protein_acc, sample_benign_gene_abbrev, sample_benign_fxn_class, sample_benign_residue, sample_benign_aa_position, sequence_dict, hydro_dict, mm_dict, cv_dict, pro_dict)
sample_pathogenic_dataframe = create_dataframe(new_sample_pathogenic_id, sample_pathogenic_protein_acc, sample_pathogenic_gene_abbrev, sample_pathogenic_fxn_class, sample_pathogenic_residue, sample_pathogenic_aa_position, sequence_dict, hydro_dict, mm_dict, cv_dict, pro_dict)
benign_dataframe = create_dataframe(new_benign_id, benign_protein_acc, benign_gene_abbrev, benign_fxn_class, benign_residue, benign_aa_position, sequence_dict, hydro_dict, mm_dict, cv_dict, pro_dict)
pathogenic_dataframe = create_dataframe(new_pathogenic_id, pathogenic_protein_acc, pathogenic_gene_abbrev, pathogenic_fxn_class, pathogenic_residue, pathogenic_aa_position, sequence_dict, hydro_dict, mm_dict, cv_dict, pro_dict)

def dropping_rows(sample_dataframe):
    filtered_data = sample_dataframe[sample_dataframe["prim_seq"].notnull()]
    filtered_data = filtered_data[filtered_data["class"].astype(str) != "synonymous-codon"]
    filtered_data = filtered_data[filtered_data["class"].astype(str) != "stop-gained"]
    filtered_data = filtered_data[filtered_data["class"].astype(str) != "stop-lost"]
    filtered_data = filtered_data[filtered_data["position"].astype(int) >= 2]
    filtered_data["prim_seq_length"] = filtered_data["prim_seq"].apply(lambda x: len(x))
    filtered_data = filtered_data[(filtered_data["position"].astype(int) + 1) <= filtered_data["prim_seq_length"]]
    return(filtered_data)
    
new_sample_benign_dataframe = dropping_rows(sample_benign_dataframe)
new_sample_pathogenic_dataframe = dropping_rows(sample_pathogenic_dataframe)
new_benign_dataframe = dropping_rows(benign_dataframe)
new_pathogenic_dataframe = dropping_rows(pathogenic_dataframe)

def generate_ref_neighbors(sample_dataframe, hydro_dict, mm_dict, cv_dict, pro_dict):
    left_neighbor = []
    right_neighbor = []
    original = []
    for row_index, row in sample_dataframe.iterrows():
        left_neighbor.append(row["prim_seq"][int(row["position"]) - 2])
        right_neighbor.append(row["prim_seq"][int(row["position"])])
        original.append(row["prim_seq"][int(row["position"]) - 1])
    sample_dataframe["left_neighbor"] = left_neighbor
    sample_dataframe["l_hydro_vals"] = sample_dataframe["left_neighbor"].map(hydro_dict)
    sample_dataframe["l_mw_vals"] = sample_dataframe["left_neighbor"].map(mm_dict)
    sample_dataframe["l_charge_vals"] = sample_dataframe["left_neighbor"].map(cv_dict)
    sample_dataframe["l_pi_vals"] = sample_dataframe["left_neighbor"].map(pi_dict)
    sample_dataframe["l_pro_pres"] = sample_dataframe["left_neighbor"].map(pro_dict)
    sample_dataframe["right_neighbor"] = right_neighbor
    sample_dataframe["r_hydro_vals"] = sample_dataframe["right_neighbor"].map(hydro_dict)
    sample_dataframe["r_mw_vals"] = sample_dataframe["right_neighbor"].map(mm_dict)
    sample_dataframe["r_charge_vals"] = sample_dataframe["right_neighbor"].map(cv_dict)
    sample_dataframe["r_pi_vals"] = sample_dataframe["right_neighbor"].map(pi_dict)
    sample_dataframe["r_pro_pres"] = sample_dataframe["right_neighbor"].map(pro_dict)
    sample_dataframe["original"] = original
    sample_dataframe["o_hydro_vals"] = sample_dataframe["original"].map(hydro_dict)
    sample_dataframe["o_mw_vals"] = sample_dataframe["original"].map(mm_dict)
    sample_dataframe["o_charge_vals"] = sample_dataframe["original"].map(cv_dict)
    sample_dataframe["o_pi_vals"] = sample_dataframe["original"].map(pi_dict)
    sample_dataframe["o_pro_pres"] = sample_dataframe["original"].map(pro_dict)
    return(sample_dataframe)

complete_sample_benign_dataframe = generate_ref_neighbors(new_sample_benign_dataframe, hydro_dict, mm_dict, cv_dict, pro_dict)
complete_sample_pathogenic_dataframe = generate_ref_neighbors(new_sample_pathogenic_dataframe, hydro_dict, mm_dict, cv_dict, pro_dict)
complete_benign_dataframe = generate_ref_neighbors(new_benign_dataframe, hydro_dict, mm_dict, cv_dict, pro_dict)
complete_pathogenic_dataframe = generate_ref_neighbors(new_pathogenic_dataframe, hydro_dict, mm_dict, cv_dict, pro_dict)

sample_benign_labels = ["0"] * len(complete_sample_benign_dataframe["class"])
sample_pathogenic_labels = ["1"] * len(complete_sample_pathogenic_dataframe["class"])
complete_sample_benign_dataframe.insert(0, "label", sample_benign_labels)
complete_sample_pathogenic_dataframe.insert(0, "label", sample_pathogenic_labels)
benign_labels = ["0"] * len(complete_benign_dataframe["class"])
pathogenic_labels = ["1"] * len(complete_pathogenic_dataframe["class"])
complete_benign_dataframe.insert(0, "label", benign_labels)
complete_pathogenic_dataframe.insert(0, "label", pathogenic_labels)

almost_final_benign = complete_benign_dataframe[["label", "prim_seq_length", "hydro_vals", "mw_vals", "charge_vals", "pi_vals", "pro_pres",
                                                "l_hydro_vals", "l_mw_vals", "l_charge_vals", "l_pi_vals", "l_pro_pres",
                                                "r_hydro_vals", "r_mw_vals", "r_charge_vals", "r_pi_vals", "r_pro_pres",
                                                "o_hydro_vals", "o_mw_vals", "o_charge_vals", "o_pi_vals", "o_pro_pres"]]
almost_final_pathogenic = complete_pathogenic_dataframe[["label", "prim_seq_length", "hydro_vals", "mw_vals", "charge_vals", "pi_vals", "pro_pres",
                                                "l_hydro_vals", "l_mw_vals", "l_charge_vals", "l_pi_vals", "l_pro_pres",
                                                "r_hydro_vals", "r_mw_vals", "r_charge_vals", "r_pi_vals", "r_pro_pres",
                                                "o_hydro_vals", "o_mw_vals", "o_charge_vals", "o_pi_vals", "o_pro_pres"]]
final_dataframe = pd.concat([almost_final_benign, almost_final_pathogenic], ignore_index = True)
#final_dataframe.plot.hist(alpha = 0.2)
#find_nan_df = final_dataframe[final_dataframe.isnull().any(axis = 1)], used to find nan values which messed up logreg

final_y = final_dataframe[["label"]].values.astype(int)
final_x = np.round(final_dataframe[["prim_seq_length", "hydro_vals", "mw_vals", "charge_vals", "pi_vals", "pro_pres",
                                 "l_hydro_vals", "l_mw_vals", "l_charge_vals", "l_pi_vals", "l_pro_pres",
                                 "r_hydro_vals", "r_mw_vals", "r_charge_vals", "r_pi_vals", "r_pro_pres",
                                 "o_hydro_vals", "o_mw_vals", "o_charge_vals", "o_pi_vals", "o_pro_pres"]].values, 2)
final_x = final_x.round(2)

sss = StratifiedShuffleSplit(test_size = 0.34, train_size = 0.66)

def cv_stratified_shuffle_split(final_y, final_x):
    sss_train_index = []
    sss_test_index = []
    for train_index, test_index in sss.split(final_x, final_y):
        sss_train_index.append(train_index)
        sss_test_index.append(test_index)
    return(sss_train_index, sss_test_index)
    
cv_sss_train, cv_sss_test = cv_stratified_shuffle_split(final_y, final_x)

# Leave One Out is too computationally/memory-expensivve for this computer
"""
loo = LeaveOneOut()

def cv_leave_one_out(final_x):
    loo_train_index = []
    loo_test_index = []
    for train_index, test_index in loo.split(final_x):
        loo_train_index.append(train_index)
        loo_test_index.append(test_index)
    return(loo_train_index, loo_test_index)

cv_loo_train, cv_loo_test = cv_leave_one_out(final_x)
"""

skf = StratifiedKFold(n_splits = 20)

def cv_stratified_k_fold(final_y, final_x):
    skf_train_index = []
    skf_test_index = []
    for train_index, test_index in skf.split(final_x, final_y):
        skf_train_index.append(train_index)
        skf_test_index.append(test_index)
    return(skf_train_index, skf_test_index)

cv_skf_train, cv_skf_test = cv_stratified_k_fold(final_y, final_x)

logreg = LogisticRegression(penalty = "l1", solver = "liblinear")

def perform_log_reg(cv_sss_train, cv_sss_test, cv_skf_train, cv_skf_test, final_y, final_x):
    tn_sss = []
    fp_sss = []
    fn_sss = []
    tp_sss = []
    tn_skf = []
    fp_skf = []
    fn_skf = []
    tp_skf = []
    for sss_iter in range(0, len(cv_sss_train)):
        logreg.fit(final_x[cv_sss_train[sss_iter]], final_y[cv_sss_train[sss_iter]].ravel())
        y_pred = logreg.predict(final_x[cv_sss_test[sss_iter]])
        y_true = final_y[cv_sss_test[sss_iter]]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        tn_sss.append(tn) 
        fp_sss.append(fp)
        fn_sss.append(fn)
        tp_sss.append(tp)
    for skf_iter in range(0, len(cv_skf_train)):
        logreg.fit(final_x[cv_skf_train[skf_iter]], final_y[cv_skf_train[skf_iter]].ravel())
        y_pred = logreg.predict(final_x[cv_skf_test[skf_iter]])
        y_true = final_y[cv_skf_test[skf_iter]]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        tn_skf.append(tn) 
        fp_skf.append(fp)
        fn_skf.append(fn)
        tp_skf.append(tp)
    return(tn_sss, fp_sss, fn_sss, tp_sss, tn_skf, fp_skf, fn_skf, tp_skf)
    
tn_sss, fp_sss, fn_sss, tp_sss, tn_skf, fp_skf, fn_skf, tp_skf = perform_log_reg(cv_sss_train, cv_sss_test, cv_skf_train, cv_skf_test, final_y, final_x)

tn_sss = np.asarray(tn_sss)
fp_sss = np.asarray(fp_sss)
fn_sss = np.asarray(fn_sss)
tp_sss = np.asarray(tp_sss)
tn_skf = np.asarray(tn_skf)
fp_skf = np.asarray(fp_skf)
fn_skf = np.asarray(fn_skf)
tp_skf = np.asarray(tp_skf)

q_sss = (np.mean(tp_sss) + np.mean(tn_sss)) / (np.mean(tp_sss) + np.mean(tn_sss) + np.mean(fn_sss) + np.mean(fp_sss))
q_skf = (np.mean(tp_skf) + np.mean(tn_skf)) / (np.mean(tp_skf) + np.mean(tn_skf) + np.mean(fn_skf) + np.mean(fp_skf))
q_sss_std = (np.std(tp_sss) + np.std(tn_sss)) / (np.std(tp_sss) + np.std(tn_sss) + np.std(fn_sss) + np.std(fp_sss))
q_skf_std = (np.std(tp_skf) + np.std(tn_skf)) / (np.std(tp_skf) + np.std(tn_skf) + np.std(fn_skf) + np.std(fp_skf))
mean_tpr_sss = np.mean(tp_sss) / (np.mean(tp_sss) + np.mean(fn_sss))
mean_tpr_skf = np.mean(tp_skf) / (np.mean(tp_skf) + np.mean(fn_skf))
mean_fpr_sss = np.mean(fp_sss) / (np.mean(fp_sss) + np.mean(tn_sss))
mean_fpr_skf = np.mean(fp_skf) / (np.mean(fp_skf) + np.mean(tn_skf))
tpr_sss = np.sort(tp_sss / (tp_sss + fn_sss))
tpr_skf = np.sort(tp_skf / (tp_skf + fn_skf))
fpr_sss = np.sort(fp_sss / (fp_sss + tn_sss))
fpr_skf = np.sort(fp_skf / (fp_skf + tn_skf))
plt.plot(tpr_sss, fpr_sss)
plt.plot(tpr_skf, fpr_skf)
plt.show()
auc_sss = integrate.simps(tpr_sss, x = fpr_sss)
auc_skf = integrate.simps(tpr_skf, x = fpr_skf)
mcc_sss = (np.mean(tp_sss) * np.mean(tn_sss) - np.mean(fp_sss) * np.mean(fn_sss)) / math.sqrt((np.mean(tp_sss) + np.mean(fn_sss)) * (np.mean(tp_sss) + np.mean(fp_sss)) * (np.mean(tn_sss) + np.mean(fn_sss)) * (np.mean(tn_sss) + np.mean(fp_sss)))
mcc_skf = (np.mean(tp_skf) * np.mean(tn_skf) - np.mean(fp_skf) * np.mean(fn_skf)) / math.sqrt((np.mean(tp_skf) + np.mean(fn_skf)) * (np.mean(tp_skf) + np.mean(fp_skf)) * (np.mean(tn_skf) + np.mean(fn_skf)) * (np.mean(tn_skf) + np.mean(fp_skf)))
mcc_sss_std = (np.std(tp_sss) * np.std(tn_sss) - np.std(fp_sss) * np.std(fn_sss)) / math.sqrt((np.std(tp_sss) + np.std(fn_sss)) * (np.std(tp_sss) + np.std(fp_sss)) * (np.std(tn_sss) + np.std(fn_sss)) * (np.std(tn_sss) + np.std(fp_sss)))
mcc_skf_std = (np.std(tp_skf) * np.std(tn_skf) - np.std(fp_skf) * np.std(fn_skf)) / math.sqrt((np.std(tp_skf) + np.std(fn_skf)) * (np.std(tp_skf) + np.std(fp_skf)) * (np.std(tn_skf) + np.std(fn_skf)) * (np.std(tn_skf) + np.std(fp_skf)))
ber_sss = 0.5 * ((np.mean(fn_sss) / (np.mean(fn_sss) + np.mean(tp_sss))) + (np.mean(fp_sss) / (np.mean(fp_sss) + np.mean(tn_sss))))
ber_skf = 0.5 * ((np.mean(fn_skf) / (np.mean(fn_skf) + np.mean(tp_skf))) + (np.mean(fp_skf) / (np.mean(fp_skf) + np.mean(tn_skf))))
ber_sss_std = 0.5 * ((np.std(fn_sss) / (np.std(fn_sss) + np.std(tp_sss))) + (np.std(fp_sss) / (np.std(fp_sss) + np.std(tn_sss))))
ber_skf_std = 0.5 * ((np.std(fn_skf) / (np.std(fn_skf) + np.std(tp_skf))) + (np.std(fp_skf) / (np.std(fp_skf) + np.std(tn_skf))))
chis_sss = len(cv_sss_test[0]) * mcc_sss ** 2
chis_skf = len(cv_skf_test[0]) * mcc_skf ** 2
chis_sss = len(cv_sss_test[0]) * mcc_sss_std ** 2
chis_skf = len(cv_skf_test[0]) * mcc_skf_std ** 2
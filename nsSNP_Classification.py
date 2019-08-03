# This project is to create a logistic regression classifier based on dbSNP data from the National Center of Biomedical Information (NCBI) and NCBI's Protein database.
# This classifier can identify whether a Single Nucleotide Polymorphism(SNP) is pathogenic (harmful) or benign (unaffective).
# Different features are utilized from the data such as....

# Import modules, packages and libraries used (numpy, matplotlib, pandas dataframe, mathematics, scikit-learn cross-validation - stratified shuffle split, stratified cross-fold), scikit-learn logistic regression and confusion matrix, scipy integration
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from textwrap import wrap

# parse dbSNP .txt data (or specific nucleotide position changes) into lists for later usage, specifically gene id, gene abbreviation, gene class, gene residue identification, amino acid change position, protein accession number
def mutation_data_parser(original_file, new_file):
    #print("Start Data Extraction")
    with open(original_file,"r") as stuff_to_write: # Open the original file as read oly
        with open(new_file,"w") as stuff_written: # Create a new data file with extracted lines from original file to preserve original file data integrity
            for line in stuff_to_write:
                if "ss_pick" not in line and "prot_acc=." not in line and "SEQ" not in line: # Extract only the desired lines with information from the original file to the new data file, specifically SNP id, sequence, protein accession lines with available data (ex: not "prot_acc=.")
                    stuff_written.write(line)
    sample_snp_id = [] # Creating empty lists to append with desired SNP info: SNP id, gene abbreviation (ex: SFTPC, etc.), SNP function class (missense, etc.), residue (amino acid) substituted, substitution position, protein accession number in NCBI database
    sample_gene_abbrev = []
    sample_fxn_class = []
    sample_residue = []
    sample_aa_position = []
    sample_protein_acc = []
    with open(new_file,"r") as sample_info:
        for line in sample_info:
            if "rs" and "Homo sapiens" in line: # Append respective list for SNP id, which has the format "rs...." usually
                sample_snp_id.append(line.split(" | ")[0])
            elif "LOC |"and "residue" in line and "fxn-class=reference" not in line: # Append respective lists for gene abbreviation, SNP function class, residue substituted, substitution position, protein accession number
                sample_gene_abbrev.append(line.split(" | ")[1])
                sample_fxn_class.append((line.split(" | ")[3])[10:])
                sample_residue.append((line.split(" | ")[6])[8:])
                sample_aa_position.append((line.split(" | ")[7])[12:])
                sample_protein_acc.append((line.split(" | ")[-1])[9:-1])
    return(sample_snp_id, sample_gene_abbrev, sample_fxn_class, sample_residue, sample_aa_position, sample_protein_acc) # Return information in tuple of lists to be used later on

sample_benign_snp_id, sample_benign_gene_abbrev, sample_benign_fxn_class, sample_benign_residue, sample_benign_aa_position, sample_benign_protein_acc = mutation_data_parser("C:/Users/Everet/Documents/CSE527/Project/Final_Report/sample_benign.txt", "C:/Users/Everet/Documents/CSE527/Project/Final_Report/sample_benign_new.txt")
sample_pathogenic_snp_id, sample_pathogenic_gene_abbrev, sample_pathogenic_fxn_class, sample_pathogenic_residue, sample_pathogenic_aa_position, sample_pathogenic_protein_acc = mutation_data_parser("C:/Users/Everet/Documents/CSE527/Project/Final_Report/sample_pathogenic.txt", "C:/Users/Everet/Documents/CSE527/Project/Final_Report/sample_pathogenic_new.txt")
benign_snp_id, benign_gene_abbrev, benign_fxn_class, benign_residue, benign_aa_position, benign_protein_acc = mutation_data_parser("C:/Users/Everet/Documents/CSE527/Project/Final_Report/Benign_nsSNP_Protein_Available.txt", "C:/Users/Everet/Documents/CSE527/Project/Final_Report/New_Benign_nsSNP_Protein_Available.txt")
pathogenic_snp_id, pathogenic_gene_abbrev, pathogenic_fxn_class, pathogenic_residue, pathogenic_aa_position, pathogenic_protein_acc = mutation_data_parser("C:/Users/Everet/Documents/CSE527/Project/Final_Report/Pathogenic_nsSNP_Protein_Available.txt", "C:/Users/Everet/Documents/CSE527/Project/Final_Report/New_Pathogenic_nsSNP_Protein_Available.txt")
    
def appending_snp_id(snp_id, gene_abbrev): # Creating a new appended SNP id list, where the format is [SNP id 0, gene abbreviation 0, SNP id 1, gene abbreviation 1, ...]
    #print("Appending SNP ID Data")
    new_snp_id = [] # Generate empty new SNP id type list
    snp_id_select = 0 # Initialized index for use on the previous snp_id lists created in the function "mutation_data_parser"
    new_snp_id.append(snp_id[0]) # Append the first entry of the list with the first SNP id
    for i_iter in range(1, len(gene_abbrev)): # Iterate from the second entry to the length of the original gene abbreviations list
        if gene_abbrev[i_iter] == gene_abbrev[i_iter - 1]: # Checks if the current gene abbreviation entry in the list is the same as the previous entry
            new_snp_id.append(snp_id[snp_id_select]) #
        elif gene_abbrev[i_iter] != gene_abbrev[i_iter - 1]: # If the current gene abbreviation entry is not the same as the previous entry, select the next SNP id from the snp_id list, and append it as the entry after the gene abbreviation
            snp_id_select += 1
            new_snp_id.append(snp_id[snp_id_select])
    return(new_snp_id)

new_sample_benign_id = appending_snp_id(sample_benign_snp_id, sample_benign_gene_abbrev)
new_sample_pathogenic_id = appending_snp_id(sample_pathogenic_snp_id, sample_pathogenic_gene_abbrev)
new_pathogenic_id = appending_snp_id(pathogenic_snp_id, pathogenic_gene_abbrev)
new_benign_id = appending_snp_id(benign_snp_id, benign_gene_abbrev)

amino_acids = ["G","A","V","L","I","P","F","Y","W","S","T","N","Q","C","M","D","E","H","K","R"] # Generate list of twenty possible amino acids in protein sequences where the SNP's occur
amino_acid_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] # Generate numeric label for the twenty possible amino acids in protein sequences where the SNP's occur, to also do a Bag-of-Words approach
hydrophobic_values = [0.67, 1.0, 2.3, 2.2, 3.1, -0.29, 2.5, 0.08, 1.5, -1.1, -0.75, -2.7, -2.9, 0.17, 1.1, -3.0, -2.6, -1.7, -4.6, -7.5] # Generate list of hydrophobicity values for each amino acid
molec_mass_values = [57, 71, 99, 113, 113, 97, 147, 163, 186, 87, 101, 114, 128, 103, 131, 115, 129, 137, 128, 156] # Generate list of molecular mass values for each amino acid
charge_values = [0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0.5, 0, -1, -1, 0.5, 1, 1] # Generate charge values for each amino acid
pI_values = [5.97, 6.01, 5.97, 5.98, 6.02, 6.48, 5.48, 5.66, 5.89, 5.68, 5.87, 5.41, 5.65, 5.07, 5.74, 2.77, 3.22, 7.59, 9.74, 10.76] # Generate pI (isoelectric point) values for each R-group (or interchangeable group) on the Amino Acids
proline_present = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # Generate values for whether or not a proline is present for a protein sequence, which causes kinks, etc.
aa_values_dict = dict(zip(amino_acids, amino_acid_values)) # Create a dictionary where the respective amino acids are assigned their respective numeric labels
hydro_dict = dict(zip(amino_acids, hydrophobic_values)) # Create a dictionary where the respective amino acids are assigned their respective hyrophobicity values
mm_dict = dict(zip(amino_acids, molec_mass_values)) # Create a dictionary where the respective amino acids are assigned their respective molecular mass values
cv_dict = dict(zip(amino_acids, charge_values)) # Create a dictionary where the respective amino acids are assigned their respective charge values
pi_dict = dict(zip(amino_acids, pI_values)) # Create a dictionary where the respective amino acids are assigned their respective pI values
pro_dict = dict(zip(amino_acids, proline_present)) # Create a dictionary where the respective amino acids are assigned their respective proline presence

def parse_all_proteins(): # This function utilizes fasta data with protein accession numbers, and creates a dictionary where protein accession number(ex: "NP_778203.1") is the key and the value of the key is the protein sequence string
    #print("Parsing Protein Data")
    sequences = {} # Initialize a blank dictionary
    in_sequence = False # Initialize a state machine that turns on ("true") later when a new FASTA protein sequence entry is being read
    current_sequence_name = "" # Initialize a blank key string name (eventually will be a protein accession number)
    with open("C:/Users/Everet/Documents/CSE527/Project/Final_Report/All_Proteins.fasta","r") as sample_fasta: # Open the all proteins FASTA file as read-only
        for line in sample_fasta.readlines():
            if line[0] == '>': # If the line in the FASTA file starts with a '>', the state machine turns on, extract the desired protein accession number as the current_sequence_name, and create a blank list of strings (sequences) for the protein accession number key
                in_sequence = True
                current_sequence_name = line.split(" ")[0].split(">")[1]
                sequences[current_sequence_name] = []
            elif line == "\n": # If the next line in the FASTA file while reading is an empty line, join all of the fragments of sequences stored in the list of strings into a full sequence for the respective protein accession number, and reset the state machine to "False" (off state) as well as protein accession number
                sequences[current_sequence_name] = "".join(sequences[current_sequence_name])
                in_sequence = False
                current_sequence_name = ""
            elif in_sequence is True: # While the state machine is "true" (on), append the line's sequence to the respective protein accession number key's list of fragment sequences
                sequences[current_sequence_name].append(line.strip())
    return(sequences)

sequence_dict = parse_all_proteins() # Create a dictionary of all the sequences from the All_Proteins.fasta file

def create_dataframe(sample_snp_id, sample_protein_acc, sample_gene_abbrev, sample_fxn_class, sample_residue, sample_aa_position, sequence_dict, hydro_dict, mm_dict, cv_dict, pro_dict): # Create a dataframe to organize all SNP entries/data/features
    #print("Creating Dataframes")
    sample_dataframe = pd.DataFrame(np.column_stack([sample_snp_id, sample_protein_acc, sample_gene_abbrev, sample_fxn_class, sample_residue, sample_aa_position]), # Create initial dataframe with SNP id, protein accession number, gene abbreviation, function class, amino acid residue substituted, and amino acid position
                                     columns = ["snp_id", "prot_acc", "gene_abb", "class", "residue", "position"])
    sample_dataframe["prim_seq"] = sample_dataframe["prot_acc"].map(sequence_dict) # Use map function to assign respective protein sequence from dictionary of all protein made to their respective protein accession number
    sample_dataframe["aa_vals"] = sample_dataframe["residue"].map(aa_values_dict)  # Use map function to assign respective hydrophobicity value to the amino acid residue substituted for the SNP
    sample_dataframe["hydro_vals"] = sample_dataframe["residue"].map(hydro_dict) # Use map function to assign respective hydrophobicity value to the amino acid residue substituted for the SNP
    sample_dataframe["mw_vals"] = sample_dataframe["residue"].map(mm_dict) # Use map function to assign respective molecular mass value to the amino acid residue substituted for the SNP
    sample_dataframe["charge_vals"] = sample_dataframe["residue"].map(cv_dict) # Use map function to assign respective charge value to the amino acid residue substituted for the SNP
    sample_dataframe["pi_vals"] = sample_dataframe["residue"].map(pi_dict) # Use map function to assign respective pI value to the amino acid residue substituted for the SNP
    sample_dataframe["pro_pres"] = sample_dataframe["residue"].map(pro_dict) # Use map function to assign respective proline presence value to the amino acid residue substituted for the SNP
    return(sample_dataframe)

sample_benign_dataframe = create_dataframe(new_sample_benign_id, sample_benign_protein_acc, sample_benign_gene_abbrev, sample_benign_fxn_class, sample_benign_residue, sample_benign_aa_position, sequence_dict, hydro_dict, mm_dict, cv_dict, pro_dict)
sample_pathogenic_dataframe = create_dataframe(new_sample_pathogenic_id, sample_pathogenic_protein_acc, sample_pathogenic_gene_abbrev, sample_pathogenic_fxn_class, sample_pathogenic_residue, sample_pathogenic_aa_position, sequence_dict, hydro_dict, mm_dict, cv_dict, pro_dict)
benign_dataframe = create_dataframe(new_benign_id, benign_protein_acc, benign_gene_abbrev, benign_fxn_class, benign_residue, benign_aa_position, sequence_dict, hydro_dict, mm_dict, cv_dict, pro_dict)
pathogenic_dataframe = create_dataframe(new_pathogenic_id, pathogenic_protein_acc, pathogenic_gene_abbrev, pathogenic_fxn_class, pathogenic_residue, pathogenic_aa_position, sequence_dict, hydro_dict, mm_dict, cv_dict, pro_dict)

def dropping_rows(sample_dataframe): # Retain desired based on different conditions
    #print("Dropping Rows")
    filtered_data = sample_dataframe[sample_dataframe["prim_seq"].notnull()] # Retain SNP rows not missing a primary protein (amino acid) sequence, since cannot use in analysis
    filtered_data = filtered_data[filtered_data["class"].astype(str) != "synonymous-codon"] # Retain SNP rows without a synonymous codon (resulting in no effect)
    filtered_data = filtered_data[filtered_data["class"].astype(str) != "stop-gained"] # Retain SNP rows without stop-gained and stop-lost (which prevent comparison to neighboring residues for the substituted one for an SNP, a key part of this investigation)
    filtered_data = filtered_data[filtered_data["class"].astype(str) != "stop-lost"]
    filtered_data = filtered_data[filtered_data["position"].astype(int) >= 2]  # Retain SNP rows where the SNP substitution is not at the first amino acid in the primary sequence (so there are both left and right neighbors to be investigated), and >= to 5 to account for alpha helix secondary bonding between resideus 3-4 amino acids away
    filtered_data["prim_seq_length"] = filtered_data["prim_seq"].apply(lambda x: len(x)) # Measure and create a oolumn in the dataframe that measures the length of the original protein's primary sequence that an SNP occurs in
    filtered_data = filtered_data[(filtered_data["position"].astype(int) + 1) <= filtered_data["prim_seq_length"]] # Retain SNP rows where the substitution is at least one less than the length of the primary sequence length (so there are both left and right neighbor residues)
    return(filtered_data)
    
new_sample_benign_dataframe = dropping_rows(sample_benign_dataframe) # Apply dropping row function to sample benign snSNP data (testing out function before applying to mass data)
new_sample_pathogenic_dataframe = dropping_rows(sample_pathogenic_dataframe) # Apply dropping row function to sample pathogenic snSNP data (testing out function before applying to mass data)
new_benign_dataframe = dropping_rows(benign_dataframe) # Apply dropping row function to full benign snSNP dataset
new_pathogenic_dataframe = dropping_rows(pathogenic_dataframe) # Apply dropping row function to full pathogenic snSNP dataset

print(new_sample_benign_dataframe)

def generate_ref_neighbors(sample_dataframe, hydro_dict, mm_dict, cv_dict, pro_dict): # Generate left and right neighboring amino acid data for the snSNP location in respective amino acid primary sequences
    #print("Generating Neighbors")
    left_neighbor = []
    right_neighbor = []
    original = []
    for row_index, row in sample_dataframe.iterrows(): # Use dataframe iterrorws() function, with row index and row series to create lists of snSNP's original amino acid, left neighboring amino acid, and right neighboring amino acid
        left_neighbor.append(row["prim_seq"][int(row["position"]) - 2]) # Append the list by accessing the specific location in original primary sequence for the protein the snSNP is applied to, and subtract 2 (due to Python zero indexing) to get the left neighbor amino acid
        right_neighbor.append(row["prim_seq"][int(row["position"])]) # Append the list by applying a similar concept as above, but the python for the right neighbor is simply the location number of the snSNP accessed in the dataframe
        original.append(row["prim_seq"][int(row["position"]) - 1]) # Append the list by applying a similar concept as for the left neighbor, but only subtract 1
    sample_dataframe["left_neighbor"] = left_neighbor # Add a column to the dataframe that represents the left neighbor amino acid of the amino acid substituted out by the snSNP
    sample_dataframe["l_aa_vals"] = sample_dataframe["left_neighbor"].map(aa_values_dict) # Add a column for the respective amino acid label value ofe the left neighbor amino acid to the dataframe
    sample_dataframe["l_hydro_vals"] = sample_dataframe["left_neighbor"].map(hydro_dict) # Add a column for the hydrophobicity value of the left neighbor amino acid to the dataframe
    sample_dataframe["l_mw_vals"] = sample_dataframe["left_neighbor"].map(mm_dict) # Add a column for the respective molecular weight value of the left neighbor amino acid to the dataframe
    sample_dataframe["l_charge_vals"] = sample_dataframe["left_neighbor"].map(cv_dict) # Add a column for the respective charge value of the left neighbor amino acid to the dataframe
    sample_dataframe["l_pi_vals"] = sample_dataframe["left_neighbor"].map(pi_dict) # Add a column for the respective pI value of the left neighbor amino acid to the dataframe
    sample_dataframe["l_pro_pres"] = sample_dataframe["left_neighbor"].map(pro_dict) # Add a column for the respective proline presence value of the left neighbor amino acid to the dataframe
    sample_dataframe["right_neighbor"] = right_neighbor # Add a column to the dataframe that represents the right neighbor amino acid of the amino acid substituted out by the snSNP
    sample_dataframe["r_aa_vals"] = sample_dataframe["right_neighbor"].map(aa_values_dict) # Add a column for the respective amino acid label of the left neighbor amino acid to the dataframe
    sample_dataframe["r_hydro_vals"] = sample_dataframe["right_neighbor"].map(hydro_dict) # Add a column for the respective hydrophobic value of the right neighbor amino acid to the dataframe
    sample_dataframe["r_mw_vals"] = sample_dataframe["right_neighbor"].map(mm_dict) # Add a column for the respective molecular weight value of the right neighbor amino acid to the dataframe
    sample_dataframe["r_charge_vals"] = sample_dataframe["right_neighbor"].map(cv_dict) # Add a column for the respective charge value of the right neighbor amino acid to the dataframe
    sample_dataframe["r_pi_vals"] = sample_dataframe["right_neighbor"].map(pi_dict) # Add a column for the respective pI value of the right neighbor amino acid to the dataframe
    sample_dataframe["r_pro_pres"] = sample_dataframe["right_neighbor"].map(pro_dict) # Add a column for the respective proline presence value of the right neighbor amino acid to the dataframe
    sample_dataframe["original"] = original # Add a column to the dataframe that represents the original amino acid substituted out by the snSNP
    sample_dataframe["o_aa_vals"] = sample_dataframe["original"].map(aa_values_dict) # Add a column for the respective amino acid label of the left neighbor amino acid to the dataframe
    sample_dataframe["o_hydro_vals"] = sample_dataframe["original"].map(hydro_dict) # Add a column for the respective hydrophobic value of the original amino acid to the dataframe
    sample_dataframe["o_mw_vals"] = sample_dataframe["original"].map(mm_dict) # Add a column for the respective molecular weight value of the original amino acid to the dataframe
    sample_dataframe["o_charge_vals"] = sample_dataframe["original"].map(cv_dict) # Add a column for the respective charge value of the original amino acid to the dataframe
    sample_dataframe["o_pi_vals"] = sample_dataframe["original"].map(pi_dict) # Add a column for the respective pI value of the original amino acid to the dataframe
    sample_dataframe["o_pro_pres"] = sample_dataframe["original"].map(pro_dict) # Add a column for the respective proline presence value of the original amino acid to the dataframe
    return(sample_dataframe)

complete_sample_benign_dataframe = generate_ref_neighbors(new_sample_benign_dataframe, hydro_dict, mm_dict, cv_dict, pro_dict)
complete_sample_pathogenic_dataframe = generate_ref_neighbors(new_sample_pathogenic_dataframe, hydro_dict, mm_dict, cv_dict, pro_dict)
complete_benign_dataframe = generate_ref_neighbors(new_benign_dataframe, hydro_dict, mm_dict, cv_dict, pro_dict)
complete_pathogenic_dataframe = generate_ref_neighbors(new_pathogenic_dataframe, hydro_dict, mm_dict, cv_dict, pro_dict)

sample_benign_labels = ["0"] * len(complete_sample_benign_dataframe["class"]) # Create binary labels for benign and pathogenic snSNP's for the sample benign dataframe, specifically 0 = benign, 1 = pathogenic
sample_pathogenic_labels = ["1"] * len(complete_sample_pathogenic_dataframe["class"]) # Create binary labels for benign and pathogenic snSNP's for the sample pathogenic dataframe, specifically 0 = benign, 1 = pathogenic
complete_sample_benign_dataframe.insert(0, "label", sample_benign_labels) # Add a column to the dataframe for benign snSNP's which all consist of 0
complete_sample_pathogenic_dataframe.insert(0, "label", sample_pathogenic_labels) # Add a column to the dataframe for pathogenic snSNP's which all consist of 1
benign_labels = ["0"] * len(complete_benign_dataframe["class"]) # Create binary labels for benign and pathogenic snSNP's for the complete benign dataframe, specifically 0 = benign, 1 = pathogenic
pathogenic_labels = ["1"] * len(complete_pathogenic_dataframe["class"]) # Create binary labels for benign and pathogenic snSNP's for the complete pathogenic dataframe, specifically 0 = benign, 1 = pathogenic
complete_benign_dataframe.insert(0, "label", benign_labels) # Add a column to the dataframe for benign snSNP's which all consist of 0
complete_pathogenic_dataframe.insert(0, "label", pathogenic_labels) # Add a column to the dataframe for pathogenic snSNP's which all consist of 1

def calculate_differences(sample_dataframe): # Create data that measures magnitudes of change in specific amino acid quantities between the snSNP substituted amino acid and original amino acid, as well as amongst left and right neighbors
    #print("Calculating Differences")
    sample_dataframe["hydro_val_change"] = (sample_dataframe["hydro_vals"] - sample_dataframe["o_hydro_vals"]).abs() # Calculate the magnitude of the hydrophobicity value change between new and original amino acid at the substituted position
    sample_dataframe["mw_val_change"] = (sample_dataframe["mw_vals"] - sample_dataframe["o_mw_vals"]).abs() # Calculate the magnitude of the molecular weight change between new and original amino acid at the substituted position
    sample_dataframe["charge_val_change"] = (sample_dataframe["charge_vals"] - sample_dataframe["o_charge_vals"]).abs() # Calculate the magnitude of the charge value change between new and original amino acid at the substituted position
    sample_dataframe["pi_val_change"] = (sample_dataframe["pi_vals"] - sample_dataframe["o_pi_vals"]).abs() # Calculate the magnitude of the pI value change between new and original amino acid at the substituted position
    sample_dataframe["pro_pres_change"] = (sample_dataframe["pro_pres"] - sample_dataframe["o_pro_pres"]).abs() # Calculate the magnitude of the proline presence change between new and original amino acid at the substituted position
    sample_dataframe["l_hydro_val_diff_change"] = ((sample_dataframe["hydro_vals"] - sample_dataframe["l_hydro_vals"]) - (sample_dataframe["o_hydro_vals"] - sample_dataframe["l_hydro_vals"])).abs() # Calculate the magnitude of the hydrophobicity value change between left neighbor and new substitute, and left neighbor and original amino acid at the substituted position
    sample_dataframe["l_mw_val_diff_change"] = ((sample_dataframe["mw_vals"] - sample_dataframe["l_mw_vals"]) - (sample_dataframe["o_mw_vals"] - sample_dataframe["l_mw_vals"])).abs() # Calculate the magnitude of the molecular weight change between left neighbor and new substitute, and left neighbor and original amino acid at the substituted position
    sample_dataframe["l_charge_val_diff_change"] = ((sample_dataframe["charge_vals"] - sample_dataframe["l_charge_vals"]) - (sample_dataframe["o_charge_vals"] - sample_dataframe["l_charge_vals"])).abs() # Calculate the magnitude of the charge value change between left neighbor and new substitute, and left neighbor and original amino acid at the substituted position
    sample_dataframe["l_pi_val_diff_change"] = ((sample_dataframe["pi_vals"] - sample_dataframe["l_pi_vals"]) - (sample_dataframe["o_pi_vals"] - sample_dataframe["l_pi_vals"])).abs() # Calculate the magnitude of the pI value change between left neighbor and new substitute, and left neighbor and original amino acid at the substituted position
    sample_dataframe["r_hydro_val_diff_change"] = ((sample_dataframe["hydro_vals"] - sample_dataframe["r_hydro_vals"]) - (sample_dataframe["o_hydro_vals"] - sample_dataframe["r_hydro_vals"])).abs() # Calculate the magnitude of the hydrophobicity value change between right neighbor and new substitute, and right neighbor and original amino acid at the substituted position
    sample_dataframe["r_mw_val_diff_change"] = ((sample_dataframe["mw_vals"] - sample_dataframe["r_mw_vals"]) - (sample_dataframe["o_mw_vals"] - sample_dataframe["r_mw_vals"])).abs() # Calculate the magnitude of the molecular weight change between right neighbor and new substitute, and right neighbor and original amino acid at the substituted position
    sample_dataframe["r_charge_val_diff_change"] = ((sample_dataframe["charge_vals"] - sample_dataframe["r_charge_vals"]) - (sample_dataframe["o_charge_vals"] - sample_dataframe["r_charge_vals"])).abs() # Calculate the magnitude of the charge value change between right neighbor and new substitute, and right neighbor and original amino acid at the substituted position
    sample_dataframe["r_pi_val_diff_change"] = ((sample_dataframe["pi_vals"] - sample_dataframe["r_pi_vals"]) - (sample_dataframe["o_pi_vals"] - sample_dataframe["r_pi_vals"])).abs() # Calculate the magnitude of the pI value change between right neighbor and new substitute, and right neighbor and original amino acid at the substituted position
    return(sample_dataframe)

benign_calculations = calculate_differences(complete_benign_dataframe)
pathogenic_calculations = calculate_differences(complete_pathogenic_dataframe)

almost_final_benign = complete_benign_dataframe[["label", "aa_vals", "hydro_val_change", "mw_val_change", "charge_val_change", "pi_val_change", "pro_pres_change", # Create a benign snSNP dataframe that captures the changes created by nsSNP's quantitatively
                                                 "l_aa_vals", "l_hydro_val_diff_change", "l_mw_val_diff_change", "l_charge_val_diff_change", "l_pi_val_diff_change", "l_pro_pres",
                                                 "r_aa_vals", "r_hydro_val_diff_change", "r_mw_val_diff_change", "r_charge_val_diff_change", "r_pi_val_diff_change", "r_pro_pres",
                                                 "o_aa_vals", "prim_seq_length"]]

almost_final_pathogenic = complete_pathogenic_dataframe[["label", "aa_vals", "hydro_val_change", "mw_val_change", "charge_val_change", "pi_val_change", "pro_pres_change", # Create a pathogenic snSNP dataframe that captures the changes created by nsSNP's quantitatively
                                                         "l_aa_vals", "l_hydro_val_diff_change", "l_mw_val_diff_change", "l_charge_val_diff_change", "l_pi_val_diff_change", "l_pro_pres",
                                                         "r_aa_vals", "r_hydro_val_diff_change", "r_mw_val_diff_change", "r_charge_val_diff_change", "r_pi_val_diff_change", "r_pro_pres",
                                                         "o_aa_vals", "prim_seq_length"]]

#print(almost_final_benign.shape, almost_final_pathogenic.shape) # Obtain dimensions of almost_final_benign, almost_final_pathogenic dataframes, to see how many data points we can randomly sample
# Since there are 32,081 benign snSNP's, and 96,367 snSNP's, we will only randomly sample 32,000 from both dataframes to create our final dataset to use

print(almost_final_benign.shape, almost_final_pathogenic.shape)

random_benign_frame = almost_final_benign.sample(n = 32080, replace = False, random_state = 1) # Randomly sample 32,000 benign snSNP data points
random_pathogenic_frame = almost_final_pathogenic.sample(n = 96360, replace = False, random_state = 1) # Randomly sample 32,000 pathogenic snSNP data points

concat_dataframe = pd.concat([random_benign_frame, random_pathogenic_frame], ignore_index = True) # Concatenate the two dataframes into a 64,000 point dataframe

final_dataframe = concat_dataframe.sample(frac = 1) # Shuffle the points amongst the dataframe randomly to prevent biased training

#find_nan_df = final_dataframe[final_dataframe.isnull().any(axis = 1)] # Used to find any NaN values which mess up logreg
#print(find_nan_df.shape) # shows that no NaN values are present

final_y = final_dataframe[["label"]].to_numpy() # Convert the binary benign and pathogenic labels to a numpy array for use in Scikit learn functions
final_x = final_dataframe[["aa_vals", "hydro_val_change", "mw_val_change", "charge_val_change", "pi_val_change", "pro_pres_change", # Convert the feature matrix of the datapoints into a numpy array
                           "l_aa_vals", "l_hydro_val_diff_change", "l_mw_val_diff_change", "l_charge_val_diff_change", "l_pi_val_diff_change", "l_pro_pres",
                           "r_aa_vals", "r_hydro_val_diff_change", "r_mw_val_diff_change", "r_charge_val_diff_change", "r_pi_val_diff_change", "r_pro_pres",
                           "o_aa_vals", "prim_seq_length"]].to_numpy().round(2)

print(final_x.shape)

#print(final_y.shape, final_x.shape) # Check if dimensions of input matrices and label array match

# Dimensionality Reduction does not need to be done to feed into the model, as this would change the dataset space, which could result in incorrect trends found

def cv_shuffle_split(cv_type, final_y, final_x): # Create a cross-validation function with the stratified shuffle split function to generate the train indices and test indices of our dataset to use in our logistic regression model
    shuffle_train_index = [] # Initialize an empty list for training set indices
    shuffle_test_index = [] # Initialize an empty list for testing set indices
    for train_index, test_index in cv_type.split(final_x, final_y): # Using the Scikit-Learn stratified shuffle split function, for each training and testing indices, append their respective lists
        shuffle_train_index.append(train_index)
        shuffle_test_index.append(test_index)
    return(shuffle_train_index, shuffle_test_index) # Return the final stratified shuffle split training and testing indices

sss = StratifiedShuffleSplit(test_size = 0.335, train_size = 0.665) # Set stratified shuffle split cross-validation test data split parameters, 34% testing and 66% training, default 10 iterations of testing

cv_sss_train, cv_sss_test = cv_shuffle_split(sss, final_y, final_x) # Use the cross-validation stratified shuffle split function on the y and x numpy arrays/matrices to generate indices for the data, 10 arrays of indices for training, 10 arrays of indicies for testing

skf = StratifiedKFold(n_splits = 20) # Set stratified k fold settings, where there will be 20 specific folds that represent ~50% of the data in each fold, default 20 iterations of testing

cv_skf_train, cv_skf_test = cv_shuffle_split(skf, final_y, final_x) # Use the cross-validation stratified K-fold split function on the y and x numpy arrays/matrices to generate indices for the data, 20 arrays of indices for training, 20 arrays of indicies for testing

"""
# Use Feature Selection after creating CV indices

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from collections import Counter
from statistics import mode

def chi2_find_best_feature_matrix(cv_shuffle_indices, final_x, final_y): # Function to find the best indices within a feature matrix after creating cross-validation indices
    cv_shuffle_p_values = [] # Initialize a blank list of p-values, which are usually >= 0 if the chi2 values are largest
    list_of_feature_amounts = []
    for iteration in range(0, len(cv_shuffle_indices)): # For each iteration in the cross-validation list of lists of indices, append the blank list with chi-squared value arrays
        cv_shuffle_p_values.append(chi2(final_x[cv_shuffle_indices[iteration]], final_y[cv_shuffle_indices[iteration]])[0])
        list_of_feature_amounts.append(sum(cv_shuffle_p_values[iteration] >= 1e3)) # Find the total amount of chi-squared values >= 1e3 amongst all of the elements in the list of chi-squared values
    best_feature_amount = mode(list_of_feature_amounts)
    test = SelectKBest(score_func = chi2, k = best_feature_amount) # Plug in the amount of most important features into the SelectKBest function
    list_of_feat_indices = [] # Initialize a blank list for the indices of the most relevant features in the feature matrix
    for iteration in range(0, len(cv_shuffle_indices)): # For each iteration, fit the SelectKBest Chi2 model to the feature matrix, transform, and append the blank list with lists of indices
        fit_sss = test.fit(final_x[cv_shuffle_indices[iteration]], final_y[cv_shuffle_indices[iteration]])
        X_new = test.transform(final_x[cv_shuffle_indices[iteration]])
        list_of_feat_indices.append(test.get_support(indices = True))
    flat_list_of_feat_indices = np.concatenate(list_of_feat_indices).ravel()
    return([i[0] for i in Counter(flat_list_of_feat_indices).most_common(best_feature_amount)]) # Return the most common indices, with amount determined by the best_feature_amount

cv_sss_feat_indices = chi2_find_best_feature_matrix(cv_sss_train, final_x, final_y)

cv_skf_feat_indices = chi2_find_best_feature_matrix(cv_skf_train, final_x, final_y)
"""

cv_sss_feat_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12, 13, 14, 15, 16, 17, 18, 19] # Model performs better without feature selction so will simply just implement this

cv_skf_feat_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12, 13, 14, 15, 16, 17, 18, 19] # Model performs better without feature selction so will simply just implement this

def perform_log_reg(cv_shuffle_train, cv_shuffle_test, cv_shuffle_features, final_y, final_x): # Perform logistic regression on the y and x numpy arrays/matrices using the different cross-validation indices
    print("Performing LogReg")
    logreg = LogisticRegression(penalty = "l2", solver = "liblinear")  # Set the logistic regression to have te L1 (least-squares regularization) penalty (due to having built-in feature selection) and liblinear solver (standard and can be used with L1)
    tn_shuffle = [] # Initialize an empty list for the true negative results when performing stratified shuffle split cross-validation with the logistic regression
    fp_shuffle = [] # Initialize an empty list for the false positive results when performing stratified shuffle split cross-validation with the logistic regression
    fn_shuffle = [] # Initialize an empty list for the false negative results when performing stratified shuffle split cross-validation with the logistic regression
    tp_shuffle = [] # Initialize an empty list for the true positive results when performing stratified shuffle split cross-validation with the logistic regression
    for shuffle_iter in range(0, len(cv_shuffle_train)): # Iterate amongst the total amount of index arrays for the stratified shuffle split list of index arrays
        logreg.fit(final_x[np.ix_(cv_shuffle_train[shuffle_iter], cv_shuffle_features)], final_y[cv_shuffle_train[shuffle_iter]].ravel()) # Fit the logistic regression model to the training data, for each individual stratified shuffle split index array, .ravel() is used to flatten the arrays to be compatible
        y_pred = logreg.predict(final_x[np.ix_(cv_shuffle_test[shuffle_iter], cv_shuffle_features)]) # Predict classification of the test data, for each individual stratified shuffle split index array
        y_true = final_y[cv_shuffle_test[shuffle_iter]] # Generate the true classification of the test data, for each individual stratified shuffle split index array
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() # Use the confusion matrix function with the current binary (0 or 1) classification predicted and true values, along with .ravel to generate the amount of true negative, false positive, false negative, and true positive results for each testing iteration
        tn_shuffle.append(tn) # Update the amount of true negatives for each testing iteration, resulting in a list of true negative amounts for each iteration
        fp_shuffle.append(fp) # Update the amount of false positives for each testing iteration, resulting in a list of false positive amounts for each iteration
        fn_shuffle.append(fn) # Update the amount of false negatives for each testing iteration, resulting in a list of false negative amounts for each iteration
        tp_shuffle.append(tp) # Update the amount of true positives for each testing iteration, resulting in a list of true positive amounts for each iteration
    return(tn_shuffle, fp_shuffle, fn_shuffle, tp_shuffle)
    
tn_sss, fp_sss, fn_sss, tp_sss = perform_log_reg(cv_sss_train, cv_sss_test, cv_sss_feat_indices, final_y, final_x) # Generate lists for true negatives, false positives, false negatives, and true positives for each iteration for each type of cross-validation

tn_skf, fp_skf, fn_skf, tp_skf = perform_log_reg(cv_skf_train, cv_skf_test, cv_skf_feat_indices, final_y, final_x) # Generate lists for true negatives, false positives, false negatives, and true positives for each iteration for each type of cross-validation

def convert_conf_matrix_to_numpy(tn_shuffle, fp_shuffle, fn_shuffle, tp_shuffle):
    print("Converting Results to Numpy Arrays")
    tn_shuffle = np.asarray(tn_shuffle) # Turn the list of true negative amounts for the 10 iterations of stratified shuffle split into a numpy array for faster processing/calculations
    fp_shuffle = np.asarray(fp_shuffle) # Turn the list of false positive amounts for the 10 iterations of stratified shuffle split into a numpy array for faster processing/calculations
    fn_shuffle = np.asarray(fn_shuffle) # Turn the list of false negative amounts for the 10 iterations of stratified shuffle split into a numpy array for faster processing/calculations
    tp_shuffle = np.asarray(tp_shuffle) # Turn the list of true positive amounts for the 10 iterations of stratified shuffle split into a numpy array for faster processing/calculations
    return(tn_shuffle, fp_shuffle, fn_shuffle, tp_shuffle)

tn_sss, fp_sss, fn_sss, tp_sss = convert_conf_matrix_to_numpy(tn_sss, fp_sss, fn_sss, tp_sss)

tn_skf, fp_skf, fn_skf, tp_skf = convert_conf_matrix_to_numpy(tn_skf, fp_skf, fn_skf, tp_skf)

def calculate_acc_prec_sens_spec(tn_data, fp_data, fn_data, tp_data):
    print("Calculating Metrics")
    acc = np.mean((tp_data + tn_data) / (tp_data + tn_data + fn_data + fp_data)).round(3) # Calculate mean of overall accuracy of model results, true negative + true positive / (true negative + true positive + false negative + false positive)
    acc_std = np.std((tp_data + tn_data) / (tp_data + tn_data + fn_data + fp_data)).round(3) # Calculate standard deviation of overall accuracy of model results
    prec = np.mean(tp_data / (tp_data + fp_data)).round(3) # Calculate mean of overall precision of model results, true positive / (true positive + false positive)
    prec_std = np.std(tp_data / (tp_data + fp_data)).round(3) # Calculate standard deviation of overall precision of model results
    sens = np.mean(tp_data / (tp_data + fn_data)).round(3) # Calculate mean of overall sensitivity of model results, true positive / (true positive + false negative)
    sens_std = np.std(tp_data / (tp_data + fn_data)).round(3) # Calculate standard deviation of overall sensitivity of model results
    spec = np.mean(tn_data / (tn_data + fp_data)).round(3) # Calculate mean of overall specificity of model results, true negative / (true negative + false positive)
    spec_std = np.std(tn_data / (tn_data + fp_data)).round(3) # Calculate standard deviation of overall specificity of model results
    return(acc, acc_std, prec, prec_std, sens, sens_std, spec, spec_std)

acc_sss, acc_sss_std, prec_sss, prec_sss_std, sens_sss, sens_sss_std, spec_sss, spec_sss_std = calculate_acc_prec_sens_spec(tn_sss, fp_sss, fn_sss, tp_sss)

acc_skf, acc_skf_std, prec_skf, prec_skf_std, sens_skf, sens_skf_std, spec_skf, spec_skf_std = calculate_acc_prec_sens_spec(tn_skf, fp_skf, fn_skf, tp_skf)


# Trying out Random Forest Classifier now NEED TO CHANGE COMMENTS

from sklearn.ensemble import RandomForestClassifier

def perform_random_forest(cv_shuffle_train, cv_shuffle_test, cv_shuffle_features, final_y,final_x):  # Perform logistic regression on the y and x numpy arrays/matrices using the different cross-validation indices
    print("Performing RF")
    rfclf = RandomForestClassifier(n_estimators = 100)
    tn_shuffle_rf = []  # Initialize an empty list for the true negative results when performing stratified shuffle split cross-validation with the logistic regression
    fp_shuffle_rf = []  # Initialize an empty list for the false positive results when performing stratified shuffle split cross-validation with the logistic regression
    fn_shuffle_rf = []  # Initialize an empty list for the false negative results when performing stratified shuffle split cross-validation with the logistic regression
    tp_shuffle_rf = []  # Initialize an empty list for the true positive results when performing stratified shuffle split cross-validation with the logistic regression
    for shuffle_iter in range(0, len(cv_shuffle_train)):  # Iterate amongst the total amount of index arrays for the stratified shuffle split list of index arrays
        rfclf.fit(final_x[np.ix_(cv_shuffle_train[shuffle_iter], cv_shuffle_features)], final_y[cv_shuffle_train[shuffle_iter]].ravel())  # Fit the logistic regression model to the training data, for each individual stratified shuffle split index array, .ravel() is used to flatten the arrays to be compatible
        y_pred = rfclf.predict(final_x[np.ix_(cv_shuffle_test[shuffle_iter], cv_shuffle_features)])  # Predict classification of the test data, for each individual stratified shuffle split index array
        y_true = final_y[cv_shuffle_test[shuffle_iter]]  # Generate the true classification of the test data, for each individual stratified shuffle split index array
        tn_rf, fp_rf, fn_rf, tp_rf = confusion_matrix(y_true, y_pred).ravel()  # Use the confusion matrix function with the current binary (0 or 1) classification predicted and true values, along with .ravel to generate the amount of true negative, false positive, false negative, and true positive results for each testing iteration
        tn_shuffle_rf.append(tn_rf)  # Update the amount of true negatives for each testing iteration, resulting in a list of true negative amounts for each iteration
        fp_shuffle_rf.append(fp_rf)  # Update the amount of false positives for each testing iteration, resulting in a list of false positive amounts for each iteration
        fn_shuffle_rf.append(fn_rf)  # Update the amount of false negatives for each testing iteration, resulting in a list of false negative amounts for each iteration
        tp_shuffle_rf.append(tp_rf)  # Update the amount of true positives for each testing iteration, resulting in a list of true positive amounts for each iteration
    tn_shuffle_rf, fp_shuffle_rf, fn_shuffle_rf, tp_shuffle_rf = convert_conf_matrix_to_numpy(tn_shuffle_rf, fp_shuffle_rf, fn_shuffle_rf, tp_shuffle_rf)
    return (tn_shuffle_rf, fp_shuffle_rf, fn_shuffle_rf, tp_shuffle_rf)

tn_sss_rf, fp_sss_rf, fn_sss_rf, tp_sss_rf = perform_random_forest(cv_sss_train, cv_sss_test, cv_sss_feat_indices, final_y, final_x)  # Generate lists for true negatives, false positives, false negatives, and true positives for each iteration for each type of cross-validation

tn_skf_rf, fp_skf_rf, fn_skf_rf, tp_skf_rf = perform_random_forest(cv_skf_train, cv_skf_test, cv_skf_feat_indices, final_y, final_x)  # Generate lists for true negatives, false positives, false negatives, and true positives for each iteration for each type of cross-validation

acc_sss_rf, acc_sss_std_rf, prec_sss_rf, prec_sss_std_rf, sens_sss_rf, sens_sss_std_rf, spec_sss_rf, spec_sss_std_rf = calculate_acc_prec_sens_spec(tn_sss_rf, fp_sss_rf, fn_sss_rf, tp_sss_rf)

acc_skf_rf, acc_skf_std_rf, prec_skf_rf, prec_skf_std_rf, sens_skf_rf, sens_skf_std_rf, spec_skf_rf, spec_skf_std_rf = calculate_acc_prec_sens_spec(tn_skf_rf, fp_skf_rf, fn_skf_rf, tp_skf_rf)

# Cannot do SVM since too many data points

plt.subplot(221)
acc_labels = ["LogReg SSS", "RF SSS", "LogReg SKF", "RF SKF"]
plt.ylabel("Model Accuracy")
plt.xlabel("Model Cross-Validation")
plt.tick_params(axis = 'x', labelsize = 6, rotation = -45)
plt.errorbar(acc_labels, np.array([acc_sss, acc_sss_rf, acc_skf, acc_skf_rf]), np.array([acc_sss_std, acc_sss_std_rf, acc_skf_std, acc_skf_std_rf]), linestyle='None', marker='^')

plt.subplot(222)
prec_labels = ["LogReg SSS", "RF SSS", "LogReg SKF", "RF SKF"]
plt.ylabel("Model Precision")
plt.xlabel("Model Cross-Validation")
plt.tick_params(axis = 'x', labelsize = 6, rotation = -45)
plt.errorbar(prec_labels, np.array([prec_sss, prec_sss_rf, prec_skf, prec_skf_rf]), np.array([prec_sss_std, prec_sss_std_rf, prec_skf_std, prec_skf_std_rf]), linestyle='None', marker='^')

plt.subplot(223)
sens_labels = ["LogReg SSS", "RF SSS", "LogReg SKF", "RF SKF"]
plt.ylabel("Sensitivity")
plt.xlabel("Model Cross-Validation")
plt.tick_params(axis = 'x', labelsize = 6, rotation = -45)
plt.errorbar(sens_labels, np.array([sens_sss, sens_sss_rf, sens_skf, sens_skf_rf]), np.array([sens_sss_std, sens_sss_std_rf, sens_skf_std, sens_skf_std_rf]), linestyle='None', marker='^')

plt.subplot(224)
spec_labels = ["LogReg SSS", "RF SSS", "LogReg SKF", "RF SKF"]
plt.ylabel("Specificity")
plt.xlabel("Model Cross-Validation")
plt.tick_params(axis = 'x', labelsize = 6, rotation = -45)
plt.errorbar(spec_labels, np.array([spec_sss, spec_sss_rf, spec_skf, spec_skf_rf]), np.array([spec_sss_std, spec_sss_std_rf, spec_skf_std, spec_skf_std_rf]), linestyle='None', marker='^')

plt.show()

# Future directions: More feature selection methods, Generate more features, integrate functions within functions so even less lines of code, comment every line, how to increase accuracy?
# Look into how to predict secondary structure interactions/molecular dynamics
# Rename File Names
# Use lambda function with dataframe to get total molecular weight of original protein based on sequence
# So far: ~95% accuracy with RF, ~77% accuracy with LogReg

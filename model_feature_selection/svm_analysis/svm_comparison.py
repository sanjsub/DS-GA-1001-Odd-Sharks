import pandas as pd
from joblib import load


dump_dir = "./svm_models/"
dump_prefix = "svm_cv_n"
clf = {}
clf_smote = {}
clf_rbf = {}
clf_rbf_smote = {}
cv_results = {}

for i in range(5, 51):
	clf[i] = load(dump_dir + dump_prefix + str(i) + "_linear.joblib")
	clf_smote[i] = load(dump_dir + dump_prefix + str(i) + "_linear_smote.joblib")
	clf_rbf[i] = load(dump_dir + dump_prefix + str(i) + "_rbf.joblib")
	clf_rbf_smote[i] = load(dump_dir + dump_prefix + str(i) + "_rbf_smote.joblib")

	# add missing columns
	cv_results_no_smote = pd.DataFrame(clf[i].cv_results_)
	cv_results_no_smote["smote"] = 0

	cv_results_smote = pd.DataFrame(clf_smote[i].cv_results_)
	cv_results_smote["param_class_weight"] = "NA"
	cv_results_smote["smote"] = 1

	cv_results_rbf_no_smote = pd.DataFrame(clf_rbf[i].cv_results_)
	cv_results_rbf_no_smote["smote"] = 0

	cv_results_rbf_smote = pd.DataFrame(clf_rbf_smote[i].cv_results_)
	cv_results_rbf_smote["param_class_weight"] = "NA"
	cv_results_rbf_smote["smote"] = 1

	cv_results[i] = pd.concat([cv_results_smote, cv_results_no_smote])
	cv_results[i]["param_kernel"] = "linear"
	cv_results[i] = pd.concat([cv_results[i], cv_results_rbf_no_smote, cv_results_rbf_smote])
	cv_results[i]["roll_num"] = i
cv_results_merged = pd.concat(cv_results.values())
print(cv_results_merged)

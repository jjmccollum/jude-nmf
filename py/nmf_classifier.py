import time #to time calculations for users
import pandas as pd #to import and export Excel spreadsheets
from sklearn.feature_extraction.text import TfidfTransformer #for reweighting the entries of the input collation matrix to isolated better-separated clusters
import scipy as sp #for solving optimization problems behind classifying lacunose witnesses
import nimfa as nf #for performing non-negative matrix factorization (NMF)

'''
Class for reading and preprocessing collation data matrices, performing rank analyses on them, and applying NMF to them.
'''
class NmfClassifier():
	min_extant = 1 #minimum number of extant readings threshold for a MS to be included for NMF
	min_support = 1 #minimum number of MSS supporting a given reading threshold for the reading to be included for NMF
	use_tfidf = False #flag indicating whether or not to use term frequency, inverse document frequency (TF-IDF) weighting on the input collation matrix; this will encourage NMF to isolate groups based on more exclusive readings
	#TODO: include user-specifiable parameter for factorization method; LSNMF is a good default, but SepNMF may also show some promise.
	#seed = 0 #seed for random number generation (included for experiment reproducibility purposes)
	extant_collation_df = None #placeholder for DataFrame containing sufficiently extant MS data (to be input to NMF)
	fragmentary_collation_df = None #placeholder for DataFrame containing insufficient MS data (to be classified in post-processing)
	rank_est_df = None #placeholder for DataFrame containing statistics from rank estimation
	W_df = None #placeholder for DataFrame containing weights for the basis matrix W (readings-profiles)
	H_df = None #placeholder for DataFrame containing weights for the mixture matrix H (profiles-MSS)
	nmf_summary_df = None #placeholder for DataFrame containing summary statistics for the latest NMF run
	fragmentary_H_df = None #placeholder for DataFrame containing weights for the mixture matrix H (profiles-MSS) of the fragmentary collation

	'''
	Constructs a new NmfClassifier with the given settings.
	'''
	def __init__(self, min_extant = 1, min_support = 1, use_tfidf = False):
		self.min_extant = min_extant
		self.min_support = min_support
		self.use_tfidf = use_tfidf
	'''
	Reads the Excel sheet at the given file address, including its row and column labels, into a Pandas DataFrame,
	removing reading rows with insufficient MS support,
	optionally applying TF-IDF weighting to the resulting collation,
	and then splitting the resulting collation into extant and fragmentary DataFrames based on user settings.
	The spreadsheet is assumed to have a header row and a header column for labels.
	'''
	def read(self, file_addr):
		#Read the raw collation table:
		print('Reading collation table into data frame (this may take a moment)...')
		t0 = time.time()
		collation_df = pd.read_excel(file_addr, index_col=0) #index_col indicates that the first column contains row labels
		print('Done in %0.4fs.' % (time.time() - t0))
		#Next, take note of all readings with MS support below the user-set threshold:
		print('Dropping readings with attestation lower than %d...' % self.min_support)
		t0 = time.time()
		rows_to_drop = []
		reading_supports = collation_df.sum(axis=1)
		for reading in reading_supports.index:
			if reading_supports[reading] < self.min_support:
				rows_to_drop.append(reading)
		print('Done in %0.4fs.' % (time.time() - t0))
		#Next, take note of all MSS extant in fewer places than the minimal threshold (we'll set them aside later):
		print('Identifying MSS with fewer than %d extant readings...' % self.min_extant)
		t0 = time.time()
		cols_to_set_aside = []
		ms_extant_readings = collation_df.sum(axis=0)
		for ms in ms_extant_readings.index:
			if ms_extant_readings[ms] < self.min_extant:
				cols_to_set_aside.append(ms)
		print('Done in %0.4fs.' % (time.time() - t0))
		#Apply TF-IDF weighting, if desired:
		if self.use_tfidf:
			print('Transforming collation to TF-IDF weighting...')
			t0 = time.time()
			mat = collation_df.values.T #get the transpose, as scikitlearn expects terms to be in columns and documents in rows
			tf_idf_mat = TfidfTransformer(norm=None, smooth_idf=False).fit_transform(mat).todense()
			collation_df = pd.DataFrame(tf_idf_mat.T, index=collation_df.index, columns=collation_df.columns) #transpose back to normal
			print('Done in %0.4fs.' % (time.time() - t0))
		#Finally, remove the sparsely-supported reading columns and split the collation into an extant set and a fragmentary set:
		collation_df = collation_df.drop(rows_to_drop, axis=0)
		self.fragmentary_collation_df = pd.DataFrame(collation_df[cols_to_set_aside])
		self.extant_collation_df = pd.DataFrame(collation_df.drop(cols_to_set_aside, axis=1))
		return
	'''
	Calculates rank estimation statistics for the given rank, using the LSNMF method.
	'''
	def get_rank_est(self, k):
		print('Running NMF rank estimation for rank %d...' % k)
		t0 = time.time()
		mat = self.extant_collation_df.values
		nmf_setup = nf.Lsnmf(mat, rank=k, seed='random', max_iter=50, n_run=500, track_factor=1)
		nmf_fit = nmf_setup()
		print('Done in %0.4fs' % (time.time() - t0))
		#Return the result as a one-row DataFrame (it will be added as part of a larger DataFrame later):
		rank_est_k_df = pd.DataFrame({'max_iter': [nmf_fit.fit.max_iter], 'n_run': [nmf_fit.fit.n_run], 'coph_cor': [nmf_fit.fit.coph_cor()], 'dispersion': [nmf_fit.fit.dispersion()]}, index = [k])
		return rank_est_k_df
	'''
	Tabulates rank estimation statistics for all ranks in the given range, using the LSNMF method.
	'''
	def get_rank_ests(self, min_k, max_k):
		print('Running NMF rank estimation for all ranks from %d to %d (this may take a few minutes)...' % (min_k, max_k))
		#Start with an empty DataFrame:
		rank_est_df = pd.DataFrame({'max_iter': [], 'n_run': [], 'coph_cor': [], 'dispersion': []}, index = [])
		for k in range(min_k, max_k + 1):
			rank_est_k_df = self.get_rank_est(k)
			rank_est_df = rank_est_df.append(rank_est_k_df)
		#Ensure that the columns stay in the expected order:
		rank_est_df = rank_est_df[['max_iter', 'n_run', 'coph_cor', 'dispersion']]
		self.rank_est_df = rank_est_df
		return
	'''
	Outputs rank estimation statistics to the given Excel file.
	'''
	def print_rank_ests(self, filename='nmf_rank_ests.xlsx'):
		#If the rank estimation statistics haven't been calculated yet, then report this and close:
		if self.rank_est_df is None:
			print('There are no rank estimation statistics available to print. Please call get_rank_ests() or get_rank_est() with the appropriate inputs first.')
			return
		print('Printing rank estimation statistics to %s...' % filename)
		t0 = time.time()
		self.rank_est_df.to_excel(filename)
		print('Done in %0.4fs' % (time.time() - t0))
		return
	'''
	Utility method that returns indexed cluster labels, given a rank k.
	'''
	def get_cluster_labels(self, k):
		cluster_labels = []
		cluster_label = ''
		for i in range(1,k+1):
			cluster_label = 'Profile ' + str(i)
			cluster_labels.append(cluster_label)
		return cluster_labels
	'''
	Applies k-rank NMF using the LSNMF method with NNDSVD initialization 
	and returns a triplet containing (1) the basis matrix W (readings-profiles) as a DataFrame, 
	(2) the mixture matrix H (profiles-MSS) as a DataFrame, 
	and (3) a DataFrame containing statistics on the NMF results.
	'''
	def get_nmf_results(self, k):
		print('Running NMF with rank k = %d...' % k)
		t0 = time.time()
		mat = self.extant_collation_df.values
		nmf_setup = nf.Lsnmf(mat, rank=k, seed='nndsvd', max_iter=32000)
		nmf_fit = nmf_setup()
		t1 = time.time()
		print('Done in %0.4fs' % (t1 - t0))
		#Get the factor matrices:
		W = nmf_fit.basis()
		H = nmf_fit.coef()
		#Convert to DataFrames, complete with their own labels:
		W_df = pd.DataFrame(W)
		H_df = pd.DataFrame(H)
		W_df.index = self.extant_collation_df.index
		W_df.columns = self.get_cluster_labels(k)
		H_df.index = self.get_cluster_labels(k)
		H_df.columns = self.extant_collation_df.columns
		nmf_summary_df = pd.DataFrame({'time (s)': [t1 - t0], 'n_iter': [nmf_fit.fit.n_iter], 'dist': [nmf_fit.fit.distance()], 'evar': [nmf_fit.fit.evar()], 'W_sparseness': [nmf_fit.fit.sparseness()[0]], 'H_sparseness': [nmf_fit.fit.sparseness()[1]]}, index = [k])
		self.W_df = W_df
		self.H_df = H_df
		self.nmf_summary_df = nmf_summary_df
		return
	'''
	Outputs the current NMF outputs, along with their summary statistics, to the given Excel file.
	'''
	def print_nmf_results(self, filename=''):
		#If the NMF results haven't been calculated yet, then report this and close:
		if self.W_df is None or self.H_df is None or self.nmf_summary_df is None:
			print('There are no NMF results to print. Please call get_nmf_results() with the appropriate inputs first.')
			return
		#Set the default filename, if one isn't specified:
		if filename == '':
			k = len(self.H_df)
			filename = './nmf_output_' + str(k) + '.xlsx'
		print('Printing results to %s...' % filename)
		t0 = time.time()
		#Open an output writer, so we can write to multiple tabs in the Excel workbook:
		writer = pd.ExcelWriter(filename)
		self.W_df.to_excel(writer, 'ReadingsBasis')
		self.H_df.to_excel(writer, 'MsCoefs')
		self.nmf_summary_df.to_excel(writer, 'Summary Statistics')
		writer.save()
		print('Done in %0.4fs' % (time.time() - t0))
		return
	'''
	Calculates the mixture matrix for the MSS in the fragmentary collation DataFrame using the current basis matrix.
	'''
	def get_fragmentary_nmf_results(self):
		#If the basis matrix hasn't been calculated yet, then report this and close:
		if self.W_df is None:
			print('The basis matrix W has not been calculated yet. Please call get_nmf_results() with the appropriate inputs first.')
			return
		W_df = self.W_df
		cluster_labels = W_df.columns
		W = sp.matrix(W_df.values) #this matrix must be converted specifically to a SciPy matrix to be input to the solver
		H_df = pd.DataFrame()
		for ms in self.fragmentary_collation_df.columns:
			col = self.fragmentary_collation_df[ms]
			h = sp.optimize.nnls(W, col)
			H_df[ms] = h[0]
			#print('Dist for MS ', ms_id, ': ', h[1])
		H_df.index = cluster_labels
		H_df.columns = H_df.columns
		self.fragmentary_H_df = H_df
		return
	'''
	Outputs the mixture matrix for the fragmentary collation to the given Excel file.
	'''
	def print_fragmentary_nmf_results(self, filename=''):
		#If the fragmentary mixture matrix hasn't been calculated yet, then report this and close:
		if self.fragmentary_H_df is None:
			print('There is no mixture matrix for the fragmentary MSS to print. Please call get_fragmentary_nmf_results() first.')
			return
		#Set the default filename, if one isn't specified:
		if filename == '':
			k = len(self.fragmentary_H_df)
			filename = './nmf_fragmentary_output_' + str(k) + '.xlsx'
		print('Printing results to %s...' % filename)
		t0 = time.time()
		self.fragmentary_H_df.to_excel(filename)
		print('Done in %0.4fs' % (time.time() - t0))
		return

'''
def getLsnmfResults(df, min_k, max_k):
    summary_filename = 'C:/Projects/jude_nmf/py/summary.xlsx'
    summary_df = pandas.DataFrame({'time (s)': [], 'n_iter': [], 'dist': [], 'evar': [], 'basis_sparseness': [], 'coef_sparseness': []}, index = [])
    for k in range(min_k, max_k):
        summary_k_df = getLsnmfResult(df, k)
        summary_df = summary_df.append(summary_k_df)
    summary_df = summary_df[['time (s)', 'n_iter', 'dist', 'evar', 'basis_sparseness', 'coef_sparseness']]
    summary_df.to_excel(summary_filename)
'''

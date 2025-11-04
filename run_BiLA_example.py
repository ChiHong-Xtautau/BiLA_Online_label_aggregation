from common import load_data
from common import majority_voting as mv
import bila_mc

def error_rate(aggregated_labels, ground_truth, method_name):
	ecount = 0
	for i in range(len(ground_truth)):
		if aggregated_labels[i] != ground_truth[i]:
			ecount += 1
	erate = float(ecount) / len(ground_truth)
	print('%s error rate:' % method_name, erate)
	return erate

def BiLA_online(noisy_labels_url, ground_truth_url, num_workers=None, \
	ieo=20, ueo=20, init_samples=500, chunk_size=50):

	print("**********************************************************")
	print(noisy_labels_url, "num_workers:", num_workers, "init_samples:", init_samples)

	# The ground truth labels are only used to verify the effectiveness of label aggregation
	noisy_labels, ground_truth, classes = load_data.load_all(noisy_labels_url, ground_truth_url)

	if not num_workers is None:
		noisy_labels = noisy_labels[:,:num_workers]
	
	# Comparing nnmc with majority voting:
	res = mv.majority_voting(noisy_labels.tolist())
	error_rate(res, ground_truth, "majority voting")

	# online learning case:
	la_model = bila_mc.BiLA_MC()
	# setting the noisy labels of some samples to init the model
	la_model.set(noisy_labels[0:init_samples], classes=classes, init_epochs=ieo, update_epochs=ueo) 
	la_model.init_train()
	batch_size = chunk_size # the batch size of active learning
	idx = 0
	num_aggregated_samples = []
	mj_all_aggregated_labels = []
	all_aggregated_labels = []
	mj_error_rates = []
	BiLA_error_rates = []
	# the samples come batch by batch. for each batch, we can obtain the aggregated labels and current confusion matrices in real-time
	while idx + batch_size < noisy_labels.shape[0]:
		print("******************************")
		print("current total samples:", idx+batch_size)
		batch_noisy_labels = noisy_labels[idx:idx+batch_size]
		num_aggregated_samples.append(idx+batch_size)

		aggregated_labels, confusion_matrices = la_model.aggregate(batch_noisy_labels)
		all_aggregated_labels.extend(aggregated_labels)
		BiLA_error_rates.append(error_rate(all_aggregated_labels, ground_truth[:idx+batch_size], "online BiLA"))

		mj_error_rates.append(error_rate(res[:idx+batch_size], ground_truth[:idx+batch_size], "mj"))
		idx += batch_size
	print("******************************")
	print("current total samples:", noisy_labels.shape[0])
	num_aggregated_samples.append(noisy_labels.shape[0])
	aggregated_labels, confusion_matrices = la_model.aggregate(noisy_labels[idx:])
	all_aggregated_labels.extend(aggregated_labels)
	BiLA_error_rates.append(error_rate(all_aggregated_labels, ground_truth, "online BiLA"))
	mj_error_rates.append(error_rate(res, ground_truth, "mj"))
	print("******************************************")
	print("num_aggregated_samples", num_aggregated_samples)
	print("BiLA_error_rates", BiLA_error_rates)
	print("mj_error_rates", mj_error_rates)
	
	la_model.close() # close the tensorflow session
	return [BiLA_error_rates[-1], mj_error_rates[-1]]


def run():
	ground_truth_file = './data/original/%s-train-labels.txt' % "cifar10"
	noisy_labels_file = './data/synthetic/ana2d-%s-train-noisy-labels-bimodal-noise0.6-empty-prop0.1.txt' % "cifar10"

	# cifar10
	BiLA_online(noisy_labels_file, ground_truth_file, ieo=20, ueo=20, init_samples=1000, chunk_size=500)
	#BiLA_online(noisy_labels_file, ground_truth_file, ieo=20, ueo=20, init_samples=1000, chunk_size=200)


if __name__ == "__main__":
	run()

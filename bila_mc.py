import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow.keras.backend as K

from common import majority_voting as mv
from common import load_data as ld

class BiLA_MC(object):
	def __init__(self):
		super(BiLA_MC, self).__init__()
		self.X = None # noisy labels of some samples for initialization
		self.classes = None # list of classes

		# training parameters
		self.learning_rate = None
		self.init_epochs = None
		self.update_epochs = None
		self.nnmc_batch_size = None

		# network parameters
		self.data_dim = None
		self.hidden_dim = None
		self.out_dim = None
		self.weights = None
		self.biases = None

		# placeholders
		self._input_data = None
		self._p_select = None
		self._pt = None

		self._pt_np = None
		self._lose_op = None
		self._train_op = None
		self._sess = None

		self.is_initialized = False
		self.is_trained = False
	
	def set(self, init_noisy_labels, classes, learning_rate=0.0001, init_epochs=20, update_epochs=20, \
		nnmc_batch_size=50, hidden_dim=[64, 32]):
		self.X = init_noisy_labels
		self.classes = classes
		self.X = ld.map_classes_to_int(self.X, self.classes)
		self.learning_rate = learning_rate
		self.init_epochs = init_epochs
		self.update_epochs = update_epochs
		self.nnmc_batch_size = nnmc_batch_size
		self.data_dim = self.X.shape[1]
		self.hidden_dim = hidden_dim
		self.out_dim = len(self.classes)

		self.__set_network_variables()


		self._input_data = tf.placeholder(tf.float32, shape=[None, self.data_dim])
		self._p_select = tf.placeholder(tf.float32, shape=[self._input_data.shape[0], len(self.classes), self.data_dim, len(self.classes)])
		self._pt = tf.placeholder(tf.float32, shape=[1, len(self.classes)])

		self.__set_label_aggregation_model()

		self.is_initialized = True
		
	@staticmethod
	def __net_init(shape):
		v =  tf.ones(shape) / tf.sqrt(shape[0] * 2.0)
		return v + tf.random_normal(shape=shape, stddev=0.05)
	
	def __pi_init(self, shape):
		pi_np = np.zeros(shape)
		X_list = self.X.tolist()
		res = mv.majority_voting(X_list)
		
		count = np.zeros([len(self.classes), self.data_dim, len(self.classes)])
		for i in range(len(X_list)):
			for k in range(self.data_dim):
				cur = X_list[i][k]
				if cur != -1:
					count[res[i]][k][cur] += 1
		for t in range(len(self.classes)):
			for k in range(self.data_dim):
				s = np.sum(count[t][k])
				for c in range(len(self.classes)):
					if s != 0:
						pi_np[t][k][c] = count[t][k][c] / s
					else:
						pi_np[t][k][c] = 1.0 / len(self.classes)
		#print(pi_np)
		return tf.convert_to_tensor(pi_np, dtype=tf.float32)

	def __pt_init(self, X=None):
		if X is None:
			X = self.X
		pt_np = np.zeros([1,len(self.classes)])
		count = [0 for c in range(len(self.classes))]
		for i in range(X.shape[0]):
			for k in range(X.shape[1]):
				if X[i][k] != -1:
					count[X[i][k]] += 1
		s = sum(count)
		for i in range(len(self.classes)):
			pt_np[0][i] = count[i] / float(s)
		self._pt_np = pt_np

	def __set_network_variables(self):
		self.weights = {
			'h1': tf.Variable(self.__net_init([self.data_dim, self.hidden_dim[0]])),
			'h2': tf.Variable(self.__net_init([self.hidden_dim[0], self.hidden_dim[1]])),
			'h3': tf.Variable(self.__net_init([self.hidden_dim[1], self.out_dim])),
			'pi_para': tf.Variable(self.__pi_init([len(self.classes), self.data_dim, len(self.classes)]))
		}
		self.biases = {
			'b1': tf.Variable(self.__net_init([self.hidden_dim[0]])),
			'b2': tf.Variable(self.__net_init([self.hidden_dim[1]])),
			'b3': tf.Variable(self.__net_init([self.out_dim]))
		}

	def __set_label_aggregation_model(self):
		# Building the encoder
		mid_data = tf.matmul(self._input_data, self.weights['h1']) + self.biases['b1']
		mid_data = tf.nn.tanh(mid_data)
		mid_data = tf.matmul(mid_data, self.weights['h2']) + self.biases['b2']
		mid_data = tf.nn.tanh(mid_data)
		mid_data = tf.matmul(mid_data, self.weights['h3']) + self.biases['b3']
		out_data = tf.nn.softmax(mid_data)

		# calc pi
		pi = tf.nn.softmax(self.weights['pi_para'])
		log_pi = tf.log(1e-10+pi)

		p_res = self._p_select * log_pi
		p_res = tf.reduce_sum(p_res, [2, 3])

		self._loss_op = self.__loss_func(out_data, p_res)

		# Using RMSPropOptimizer provided by tensorflow to implement the optimizer of this paper
		# Constant value bounds are used here
		optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
		gvs = optimizer.compute_gradients(self._loss_op)
		capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
		self._train_op = optimizer.apply_gradients(capped_gvs)

	# Define the Loss
	def __loss_func(self, out_data, log_p_lt):
		temp = 0.1 * out_data * tf.log(1e-10+(self._pt / (out_data + 1e-10)))
		temp = tf.reduce_sum(-temp, 1)
		temp = temp - tf.reduce_sum(log_p_lt * out_data, 1)
		return tf.reduce_mean(temp)

	# Using the noisy labels of some samples to initialize the label aggregation model
	def init_train(self):
		if not self.is_initialized:
			print("Please initialize the object using the set function.")
			return

		# Initialize the variables (i.e. assign their default value)
		init = tf.global_variables_initializer()
		gpu_options = tf.GPUOptions(allow_growth=True)
		self._sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

		# Run the initializer
		self._sess.run(init)

		self.__pt_init()

		batch_size = self.nnmc_batch_size
		for i in range(self.init_epochs):
			times = int(self.X.shape[0] / batch_size)
			if times * batch_size < self.X.shape[0]:
				times += 1
			for c in range(1, times+1):
				batch_x = None
				if c != times:
					batch_x = self.X[(c-1)*batch_size:c*batch_size,:]
				else:
					batch_x = self.X[(c-1)*batch_size:,:]

				p_select_np = np.zeros([batch_x.shape[0], len(self.classes), self.data_dim, len(self.classes)])
				for j in range(batch_x.shape[0]):
					for t in range(len(self.classes)):
						for k in range(batch_x.shape[1]):
							cur = batch_x[j][k]
							if cur != -1:
								p_select_np[j][t][k][cur] = 1.0

				# Train
				feed_dict = {self._input_data: batch_x, self._p_select: p_select_np, self._pt:self._pt_np}
				_, l = self._sess.run([self._train_op, self._loss_op], feed_dict=feed_dict)

		self.is_trained = True

	# using the noisy labels of a batch to update the label aggregation model
	# Note: this is a private function. it's called by the aggregate function.
	def __update_train(self, noisy_labels_batch):
		batch_size = self.nnmc_batch_size
		X = noisy_labels_batch
		self.__pt_init(X)
		for i in range(self.update_epochs):
			times = int(X.shape[0] / batch_size)
			if times * batch_size < X.shape[0]:
				times += 1
			for c in range(1, times+1):
				batch_x = None
				if c != times:
					batch_x = X[(c-1)*batch_size:c*batch_size,:]
				else:
					batch_x = X[(c-1)*batch_size:,:]

				p_select_np = np.zeros([batch_x.shape[0], len(self.classes), self.data_dim, len(self.classes)])
				for j in range(batch_x.shape[0]):
					for t in range(len(self.classes)):
						for k in range(batch_x.shape[1]):
							cur = batch_x[j][k]
							if cur != -1:
								p_select_np[j][t][k][cur] = 1.0

				# update
				feed_dict = {self._input_data: batch_x, self._p_select: p_select_np, self._pt:self._pt_np}
				_, l = self._sess.run([self._train_op, self._loss_op], feed_dict=feed_dict)


	# inputting the noisy labels of a batch of samples
	# outputs: aggregated labels, current confusion matrices
	def aggregate(self, noisy_labels_batch):
		if not self.is_trained:
			print("Please run init_train before using this function.")
			return

		noisy_labels_batch = ld.map_classes_to_int(noisy_labels_batch, self.classes)
		self.__update_train(noisy_labels_batch)

		pi = tf.nn.softmax(self.weights['pi_para'])
		pi_res = self._sess.run(pi)

		# demensions k, t, c, which is the same as the proposal
		confusion_matrices = np.zeros([self.data_dim, len(self.classes), len(self.classes)])
		# constructing confusion matrices
		for k in range(self.data_dim):
			confusion_matrices[k] = pi_res[:,k,:]

		predict_res = []
		X = noisy_labels_batch
		for i in range(X.shape[0]):
			pp = [1.0 for c in range(len(self.classes))]
			for t in range(len(self.classes)):
				for j in range(self.data_dim):
					cur = X[i][j]
					if cur != -1:
						pp[t] = pp[t] * confusion_matrices[j][t][cur]
			predict_res.append(self.classes[pp.index(max(pp))])
		return predict_res, confusion_matrices 
	
	def close(self):
		self._sess.close()

		self._input_data = None
		self._p_select = None
		self._pt = None

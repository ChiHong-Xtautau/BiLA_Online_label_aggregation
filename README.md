# BiLA: Online Label Aggregation
This is the code of the BiLA framework which is published on WWW2021. 

If the code helps your work, please cite the following:

"Hong, C., Ghiassi, A., Zhou, Y., Birke, R. and Chen, L.Y., 2021, April. Online label aggregation: A variational bayesian approach. In Proceedings of the Web Conference 2021 (pp. 1904-1915)."

An example of using the label aggregation function is shown in "run_BiLA_example.py".

# To run this file
The project is developed under python 3.8.10. If you use a newer version of Python, some dependency packages may fail to install. You might see warnings when running the code because TensorFlow and its dependencies have been updated multiple times since the code was completed. However, these warnings will not affect the correctness of the algorithm.

- pip install -r requirements.txt
- python run_BiLA_example.py

# Configuration
- bila_mc.BiLA_MC is the class of the label aggregation algorithm.

# Expected Results
- The error rate on the online aggregated labels of BiLA and majority voting will be shown after running.
- one training log is shown in "./example_log.txt"
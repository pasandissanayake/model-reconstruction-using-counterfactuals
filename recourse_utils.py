import choix
import numpy as np 
from sklearn.linear_model import LogisticRegression, LinearRegression
import lime
import lime.lime_tabular
from sklearn.base import BaseEstimator, TransformerMixin

#Find indices where reocourse is needed
def recourse_needed(predict_fn, X, target=1):
	return np.where(predict_fn(X) == 1-target)[0]

#Recourse validity
def recourse_validity(predict_fn, rs,siz, target=1):
	rs=np.array(rs)
	rs=rs.reshape((-1, siz))	
	return sum(np.argmax(predict_fn(rs), axis=1)==target)/len(rs)

#Simulated pairwise feature costs
class PFC():
	def __init__(self, n_feat, n_cmps=100, seed=0):
		self.n_feat = n_feat
		self.n_cmps = n_cmps
		self.seed = seed

	def gen_feat_cmps(self):
		np.random.seed(self.seed)
		cmps = []
		for i in range(self.n_feat):
			for j in range(self.n_feat):
				if i!=j:
					for _ in range(int(self.n_cmps/2)):
						if np.random.uniform()<0.5:
							cmps.append((i,j))
						else:
							cmps.append((j,i))
		return cmps

	def get_costs(self):
		feat_cmps = self.gen_feat_cmps()
		feature_costs = choix.ilsr_pairwise(n_items=self.n_feat, data=feat_cmps, alpha=0.01)
		feature_costs = feature_costs-min(feature_costs) #shift to >=
		return feature_costs

#Functions to compute the cost of recourses
def l1_cost(xs, rs):
	cost = []
	for x,r in zip(xs,rs):
		cost.append(np.linalg.norm(r-x,1))
	return np.mean(np.array(cost))

def pfc_cost(xs, rs, feature_costs):
	costs = []
	for x,r in zip(xs,rs):
		cost = np.matmul(feature_costs, np.abs(r-x))
		costs.append(cost)
	return np.mean(np.array(costs))

#LIME wrapper for non-linear base models
def lime_explanation(model_pred_proba, X_train, x):
	explainer = lime.lime_tabular.LimeTabularExplainer(
		training_data = X_train,
		feature_selection='none',
		discretize_continuous=False
	)
	exp = explainer.explain_instance(data_row=x,
						predict_fn=model_pred_proba,
						labels=(1,),
						num_samples=2000,
						model_regressor=LogisticRegression())
	coefficients = exp.local_exp[1][0][1]
	intercept = exp.intercept[1]
	return coefficients, intercept


# def lime_explanation(model_pred_proba, X_train, x, cat_feats=None, labels=(1,)):
#     explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train,
#                                                        categorical_features=cat_feats,
#                                                        discretize_continuous=False,
#                                                        feature_selection='none')
#     exp = explainer.explain_instance(x,
#                                      model_pred_proba,
#                                      num_features=X_train.shape[1],
#                                      model_regressor=LogisticRegression(), num_samples=20000, labels=labels)
#                                      #num_samples=20000,
#                                      #labels=(1,))
#     #coefficients = exp.local_exp[0][0][1]
#     coefficients = exp.local_exp[labels[0]][0][1]
#     # the first key should be 1 if we are using lime to approximate the clf's behaviours for predicting label 1
#     intercept = exp.intercept[labels[0]]
#     #intercept = exp.intercept[0]
#     return coefficients, intercept

class GermanSCM():
	def __init__(self, X):
		self.f3 = LinearRegression()
		self.f4 = LinearRegression()
		self.personal_status_sex_cols = [c for c in list(X) if "personal_status_sex" in c]
		self.f3.fit(X[self.personal_status_sex_cols+['age']], X['amount'])
		self.f4.fit(X[['amount']], X['duration'])
		self.idx_map = {c:i for i,c in enumerate(list(X))}
		

	def act(self, x_og, grad):
		rec = np.zeros(len(x_og))

		x_og = x_og.flatten()

		u1 = x_og[[self.idx_map[c] for c in self.personal_status_sex_cols]]
		u2 = x_og[self.idx_map["age"]]
		u3 = x_og[self.idx_map["amount"]] - self.f3.predict(
			[x_og[[self.idx_map[c] for c in self.personal_status_sex_cols]+[self.idx_map["age"]]]])[0]
		u4 = x_og[self.idx_map["duration"]] - self.f4.predict([[x_og[self.idx_map["age"]]]])[0]
		
		grad = grad.flatten()

		#u1 is immutable so a1 is not actionable
		a2 = x_og[self.idx_map["age"]]+grad[self.idx_map["age"]]
		a3 = x_og[self.idx_map["amount"]]+grad[self.idx_map["amount"]]
		a4 = x_og[self.idx_map["duration"]]+grad[self.idx_map["duration"]]
		
		x1 = u1 
		if grad[self.idx_map["age"]]>0:
			x2 = a2
		else:
			x2 = u2
		x3 = a3
		x4 = a4

		rec[[self.idx_map[c] for c in self.personal_status_sex_cols]] = x1
		rec[self.idx_map["age"]] = x2
		rec[self.idx_map["amount"]] = x3
		rec[self.idx_map["duration"]] = x4

		return rec 

class SimDataSCM():
	def __init__(self, X):
		self.f2 = LinearRegression()
		self.f2.fit(X[:,0], X[:,1])
		

	def act(self, x_og, grad):
		rec = np.zeros(len(x_og))

		x_og = x_og.flatten()

		u1 = x_og[0]
		u2 = x_og[1]-self.f2.predict(x_og[0])[0]
		
		grad = grad.flatten()

		a1 = x_og[0]+grad[0]
		a2 = x_og[1]+grad[1]
		
		x1 = a1 
		x2 = a2

		rec[0] = x1
		rec[1] = x2

		return rec 

class DummyScaler(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass

	def fit(self):
		pass

	def transform(self, X):
		return X




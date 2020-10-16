from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeCV, SGDRegressor, \
     ElasticNet, ElasticNetCV, Lars, LarsCV, Lasso, LassoCV, LassoLars, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor, VotingRegressor
models = {
          'regression': {
               'linear_models': {
                    'LinearRegression': LinearRegression,
                    'LogisticRegression': LogisticRegression,
                    'Ridge': Ridge,
                    'RidgeCV': RidgeCV,
                    'SGDRegressor': SGDRegressor,
                    'ElasticNet': ElasticNet,
                    'ElasticNetCV': ElasticNetCV,
                    'Lars': Lars,
                    'LarsCV': LarsCV,
                    'Lasso': Lasso,
                    'LassoCV': LassoCV,
                    'LassoLars': LassoLars,
                    'HuberRegressor': HuberRegressor
                    },

                'ensemble': {
                    'RandomForestRegressor': RandomForestRegressor,
                    'AdaBoostRegressor': AdaBoostRegressor,
                    'BaggingRegressor': BaggingRegressor,
                    'GradientBoostingRegressor': GradientBoostingRegressor,
                    #'VotingRegressor': VotingRegressor()
                }


          }
            
           
            
}
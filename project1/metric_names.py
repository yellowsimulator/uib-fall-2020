from sklearn.metrics import r2_score, mean_squared_error

metrics = {
    'regression': {
        'r2': r2_score,
        'mse': mean_squared_error
    },
    
    'classification': {

    }
}
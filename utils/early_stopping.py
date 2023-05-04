import numpy as np

class EarlyStopping:

    """
    Early Stopping Class
    """

    def __init__(self, 
                 patience: int = 5, 
                 min_delta: float = 0.0, 
                 best_score: float = None, 
                 counter: int = 0):
        
        """ Early Stopping Class Initialiser

        Parameters:
        -----------
        patience : int
            Patience for early stopping
        min_delta : float
            Minimum delta for early stopping
        best_score : float
            Best score for early stopping
        counter : int
            Counter for early stopping

        Returns:
        --------
        None

        """

        self.patience = patience
        self.counter = counter
        self.best_score = best_score
        self.early_stop = False
        self.min_delta = min_delta

    def __call__(self, 
                 validation_loss: float, 
                 counter_overwrite: bool = False) -> tuple:

        """ Early Stopping Class Call Function

        Parameters:
        ----------- 
        validation_loss : float
            Validation loss
        counter_overwrite : bool
            Counter overwrite

        Returns:
        --------
        early_stop : bool
            Early stopping boolean
        best_score : float
            Best score
        counter : int
            Counter

        """

        score = validation_loss

        if counter_overwrite == True:
            self.counter = 0
            self.best_score = None

        if self.best_score is None:
            self.best_score = score
        
        elif np.greater_equal(self.min_delta, self.best_score - score):
            self.counter += 1
            print("Early Stopping Counter: {}/{}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop, self.best_score, self.counter
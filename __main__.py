from spam_hunter import SpamHunter
import numpy as np

if __name__ == "__main__":
    # Use SpamHunter('Euron-spam', 40, 1, False) to process data again
    hunter = SpamHunter('Euron-spam', 40, 6, True)
    [linear_svc_model, linear_test_result] = hunter.train_linear_svc()
    [multinomial_nb_model, multi_nb_test_result] = hunter.train_multinomial_nb()
    [gaussian_nb_model, gaussian_nb_test_result] = hunter.train_gaussian_nb()
    [nusvc_model, nusvc_test_result] = hunter.train_nusvc()
    [scv_model, svc_test_result] = hunter.train_svc()

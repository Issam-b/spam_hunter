from spam_hunter import SpamHunter
import numpy as np

if __name__ == "__main__":
    hunter = SpamHunter('Euron-spam', 40, 6)
    [linear_svc_model, linear_test_result] = hunter.train_linear_svc()
    [multinomial_nb_model, multi_nb_test_result] = hunter.train_multinomial_nb()
    [nusvc_model, nusvc_test_result] = hunter.train_nusvc()
    [gaussian_nb_model, gaussian_nb_test_result] = hunter.train_gaussian_nb()
    [scv_model, svc_test_result] = hunter.train_svc()

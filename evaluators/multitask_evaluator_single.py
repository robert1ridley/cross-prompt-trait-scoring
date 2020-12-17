from metrics.metrics import *
from utils.general_utils import rescale_single_attribute


class Evaluator():

    def __init__(self, test_prompt_id, X_dev_prompt_ids, X_test_prompt_ids, dev_features_list, test_features_list,
                 Y_dev, Y_test, attribute_name):
        self.attribute_name = attribute_name
        self.test_prompt_id = test_prompt_id
        self.dev_features_list = dev_features_list
        self.test_features_list = test_features_list
        self.X_dev_prompt_ids, self.X_test_prompt_ids = X_dev_prompt_ids, X_test_prompt_ids
        self.Y_dev, self.Y_test = Y_dev, Y_test
        self.Y_dev_org = Y_dev.flatten() * 100
        Y_test_flat = Y_test.flatten()
        self.Y_test_org = rescale_single_attribute(Y_test_flat, self.X_test_prompt_ids, attribute_name)
        self.best_dev = [-1, -1, -1, -1]
        self.best_test = [-1, -1, -1, -1]

    def calc_correl(self, dev_pred, test_pred):
        self.dev_pr = pearson(self.Y_dev_org, dev_pred)
        self.test_pr = pearson(self.Y_test_org, test_pred)

        self.dev_spr = spearman(self.Y_dev_org, dev_pred)
        self.test_spr = spearman(self.Y_test_org, test_pred)

    def calc_kappa(self, dev_pred, test_pred, weight='quadratic'):
        self.dev_qwk = kappa(self.Y_dev_org, dev_pred, weight)
        self.test_qwk = kappa(self.Y_test_org, test_pred, weight)

    def calc_rmse(self, dev_pred, test_pred):
        self.dev_rmse = root_mean_square_error(self.Y_dev_org, dev_pred)
        self.test_rmse = root_mean_square_error(self.Y_test_org, test_pred)

    def evaluate(self, model, epoch, print_info=True):
        dev_pred = model.predict(self.dev_features_list, batch_size=32)
        test_pred = model.predict(self.test_features_list, batch_size=32)

        dev_pred_int = dev_pred.flatten() * 100
        test_pred_flat = test_pred.flatten()
        test_pred_int = rescale_single_attribute(test_pred_flat, self.X_test_prompt_ids, self.attribute_name)
        test_pred_int = test_pred_int.flatten()

        self.calc_correl(dev_pred_int, test_pred_int)
        self.calc_kappa(dev_pred_int, test_pred_int)
        self.calc_rmse(dev_pred_int, test_pred_int)

        if self.dev_qwk > self.best_dev[0]:
            self.best_dev = [self.dev_qwk, self.dev_pr, self.dev_spr, self.dev_rmse]
            self.best_test = [self.test_qwk, self.test_pr, self.test_spr, self.test_rmse]
            self.best_dev_epoch = epoch
        if print_info:
            self.print_info()

    def print_info(self):
        print('Prompt: {}, Attribute: {}'.format(self.test_prompt_id, self.attribute_name))
        print('[DEV]   QWK:  %.3f, PRS: %.3f, SPR: %.3f, RMSE: %.3f, (Best @ %i: {{%.3f}}, %.3f, %.3f, %.3f)' % (
            self.dev_qwk, self.dev_pr, self.dev_spr, self.dev_rmse, self.best_dev_epoch,
            self.best_dev[0], self.best_dev[1], self.best_dev[2], self.best_dev[3]))
        print('[TEST]  QWK:  %.3f, PRS: %.3f, SPR: %.3f, RMSE: %.3f (Best @ %i: {{%.3f}}, %.3f, %.3f, %.3f)' % (
            self.test_qwk, self.test_pr, self.test_spr, self.test_rmse, self.best_dev_epoch,
            self.best_test[0], self.best_test[1], self.best_test[2], self.best_test[3]))

        print(
            '--------------------------------------------------------------------------------------------------------------------------')

    def print_final_info(self):
        print(
            '--------------------------------------------------------------------------------------------------------------------------')
        # print('Missed @ Epoch %i:' % self.best_test_missed_epoch)
        # print('  [TEST] QWK: %.3f' % self.best_test_missed)
        print('Prompt: {}, Attribute: {}'.format(self.test_prompt_id, self.attribute_name))
        print('Best @ Epoch %i:' % self.best_dev_epoch)
        print('  [DEV]  QWK: %.3f,  PRS: %.3f, SPR: %.3f, RMSE: %.3f' % (
        self.best_dev[0], self.best_dev[1], self.best_dev[2], self.best_dev[3]))
        print('  [TEST] QWK: %.3f,  PRS: %.3f, SPR: %.3f, RMSE: %.3f' % (
        self.best_test[0], self.best_test[1], self.best_test[2], self.best_test[3]))

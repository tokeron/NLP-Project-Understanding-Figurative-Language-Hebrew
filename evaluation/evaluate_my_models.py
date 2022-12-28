import os.path

from datasets import load_dataset
from sklearn.metrics import plot_confusion_matrix

from evaluation.evaluate_models import *

from config.config_parser import *
import wandb


def evaluate_all_models(checkpoint_list, results_path, do_eval, plot_results=False):
    """
    Evaluates all models in the checkpoint list.
    :param checkpoint_list: list of checkpoint paths
    :param results_path: path to save the results
    """

    if len(checkpoint_list) == 0:
        print("No checkpoints found.")
        return

    main_path = results_path

    # Initialize the result dataframe
    df = pd.DataFrame(columns=['checkpoint', 'dataset', 'split', 'accuracy', 'precision', 'recall', 'f1',
                               'confusion_matrix'
                               'accuracy_seen', 'precision_seen', 'recall_seen', 'f1_seen', 'confusion_matrix_seen',
                               'accuracy_unseen', 'precision_unseen', 'recall_unseen', 'f1_unseen',
                               'confusion_matrix_unseen'
                                # 'sklearn_accuracy', 'sklearn_precision', 'sklearn_recall',
                                # 'sklearn_f1', 'sklearn_confusion_matrix'
                    ])

    # if do_eval:
    for checkpoint in checkpoint_list:
        if checkpoint.split('/')[-1].startswith('wordbert'):
            dataset_name = '/home/tok/figurative-language/datasets/PerWordRowsDataset.py'
            raw_dataset = load_dataset(dataset_name)
            train_dataset = raw_dataset['train']
            test_dataset = raw_dataset['test']
            validation_dataset = raw_dataset['validation']
        else:
            dataset_name = '/home/tok/figurative-language/datasets/MetaphorRowsDataset.py'
            raw_dataset = load_dataset(dataset_name)
            train_dataset = raw_dataset['train']
            test_dataset = raw_dataset['test']
            validation_dataset = raw_dataset['validation']

        checkpoint_name = checkpoint.split('/')[-2] if checkpoint.split('/')[-1].startswith('checkpoint') else \
            checkpoint.split('/')[-1]
        print("Evaluating " + dataset_name + " with " + checkpoint)
        results_path = os.path.join(main_path, checkpoint_name)
        for dataset_for_test in [test_dataset, validation_dataset]:
            res = model_evaluation(checkpoint, train_dataset, dataset_for_test, dataset_name, results_path,
                                   dataset_split='test' if dataset_for_test == test_dataset else 'validation')
            # Add the results to the dataframe.
            split = 'test' if dataset_for_test == test_dataset else 'validation'
            res_for_df = {'checkpoint': checkpoint_name, 'dataset': dataset_name, 'split': split,
                          'accuracy': res['total']['accuracy'], 'precision': res['total']['precision'],
                          'recall': res['total']['recall'], 'f1': res['total']['f1'],
                          'confusion_matrix': res['total']['cf_matrix'],
                          'accuracy_seen': res['seen']['accuracy'], 'precision_seen': res['seen']['precision'],
                          'recall_seen': res['seen']['recall'], 'f1_seen': res['seen']['f1'],
                          'confusion_matrix_seen': res['seen']['cf_matrix'],
                          'accuracy_unseen': res['unseen']['accuracy'], 'precision_unseen': res['unseen']['precision'],
                          'recall_unseen': res['unseen']['recall'], 'f1_unseen': res['unseen']['f1'],
                          'confusion_matrix_unseen': res['unseen']['cf_matrix']
                          # 'sklearn_accuracy': res['sklearn_results']['accuracy'],
                          # 'sklearn_precision': res['sklearn_results']['precision'],
                          # 'sklearn_recall': res['sklearn_results']['recall'],
                          # 'sklearn_f1': res['sklearn_results']['f1'],
                          # 'sklearn_confusion_matrix': res['sklearn_results']['cf_matrix'],
                          # 'sklearn_classification_report': res['sklearn_results']['classification_report']
                          }
            # convert dictionary to dataframe
            res_df = pd.DataFrame(columns=res_for_df.keys(), data=[res_for_df.values()])
            wandb.log({"all_models_evaluated_{}".format(split): wandb.Table(dataframe=res_df)})
            df = df.append(res_for_df, ignore_index=True)
    # save to csv with date and time
    # wandb.Table(dataframe=df)
    df.to_csv("/home/tok/figurative-language/results/eval_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv", index=False)
    # save pickle
    df.to_pickle("/home/tok/figurative-language/results/all_models_evaluated-{}.pkl".format(
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    print("Done")

    # else:
    #     # Load the dataframe
    #     df = pd.read_csv("/home/tok/figurative-language/results/all_models_evaluated_{}.csv".format(
    #         datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    # report the results to wandb

    # # create a copy of df with split == validation
    # df_validation = df[df['split'] == 'test']
    #
    # results_folder_name = main_path.split('/')[-1]
    # plots_path = os.path.join(main_path, '..', 'plots_{}'.format(results_folder_name))
    # if not os.path.exists(plots_path):
    #     os.makedirs(plots_path)

    # # Print to best_models.txt the names of the models best performed on each dataset
    # with open("{}/best_models.txt".format(plots_path), "w") as f:
    #     datasets = 'PerWordParaDataset'
    #     split = dataset[1][1]
    #     dataset = dataset[1][0]
    #     f.write("Best models on " + dataset + " " + split + ":\n")
    #     df_curr_dataset_sorted = df_validation[df_validation['dataset'] == dataset].sort_values(by=['f1'], ascending=False)
    #     for k in range(df_curr_dataset_sorted.shape[0]):
    #         f.write("Number " + str(k + 1) + ": " + df_curr_dataset_sorted.iloc[k]['checkpoint'] + " with f1 = " +
    #                 str(df_curr_dataset_sorted.iloc[k]['f1']) + "and accuracy = "
    #                 + str(df_curr_dataset_sorted.iloc[k]['accuracy']) + "\n")
    #     f.write("\n")
    #     wandb.log({"best_models_on_" + dataset: df_curr_dataset_sorted})

    if plot_results:
        plots_path = os.path.join(results_path, 'plots')
        top_k = 10
        # Explore the results
        # Results by accuracy
        df['checkpoint'] = df['checkpoint'].astype(str) + '_' + df['split'].astype(str)
        df.sort_values(by=['accuracy'], ascending=False, inplace=True)
        df.head(top_k).plot(kind='barh', x='checkpoint', y='accuracy', legend=False, figsize=(15, 10))
        # informative axis labels
        plt.xlabel('Accuracy')
        plt.ylabel('model')
        # show the values on the bars (two decimal places)
        for i, v in enumerate(df.head(top_k)['accuracy']):
            plt.text(v + 0.001, i, '%.2f' % v, color='black')
        # Add a title
        plt.title('Top ' + str(top_k) + ' models by accuracy on test set')
        plot_name = 'accuracy_by_model_acc_top_' + str(top_k) + '.png'
        plt.savefig(os.path.join(plots_path, plot_name))
        wandb.log({"accuracy_by_model_acc_top_" + str(top_k): wandb.Image(os.path.join(plots_path, plot_name))})
        plt.show()

        # Results by f1 score
        df.sort_values(by=['f1'], ascending=False, inplace=True)
        df.head(top_k).plot(kind='barh', x='checkpoint', y='f1', legend=False, figsize=(15, 10))
        # informative axis labels
        plt.xlabel('F1')
        plt.ylabel('model')
        # show the values on the bars (two decimal places)
        for i, v in enumerate(df.head(top_k)['f1']):
            plt.text(v + 0.001, i, '%.2f' % v, color='black')
        # Add a title
        plt.title('Top ' + str(top_k) + ' models by F1 score on test set')
        plot_name = 'f1_by_model_f1_top_' + str(top_k) + '.png'
        plt.savefig(os.path.join(plots_path, plot_name))
        wandb.log({"f1_by_model_f1_top_" + str(top_k): wandb.Image(os.path.join(plots_path, plot_name))})
        plt.show()

        # Results by precision
        df.sort_values(by=['precision'], ascending=False, inplace=True)
        df.head(top_k).plot(kind='barh', x='checkpoint', y='precision', legend=False, figsize=(15, 10))
        # informative axis labels
        plt.xlabel('Precision')
        plt.ylabel('model')
        # show the values on the bars (two decimal places)
        for i, v in enumerate(df.head(top_k)['precision']):
            plt.text(v + 0.001, i, '%.2f' % v, color='black')
        # Add a title
        plt.title('Top ' + str(top_k) + ' models by precision on test set')
        plot_name = 'precision_by_model_prec_top_' + str(top_k) + '.png'
        plt.savefig(os.path.join(plots_path, plot_name))
        wandb.log({"precision_by_model_prec_top_" + str(top_k): wandb.Image(os.path.join(plots_path, plot_name))})
        plt.show()

        # Results by recall
        df.sort_values(by=['recall'], ascending=False, inplace=True)
        df.head(top_k).plot(kind='barh', x='checkpoint', y='recall', legend=False, figsize=(15, 10))
        # informative axis labels
        plt.xlabel('Recall')
        plt.ylabel('model')
        # show the values on the bars (two decimal places)
        for i, v in enumerate(df.head(top_k)['recall']):
            plt.text(v + 0.001, i, '%.2f' % v, color='black')
        # Add a title
        plt.title('Top ' + str(top_k) + ' models by recall on test set')
        plot_name = 'recall_by_model_recall_top_' + str(top_k) + '.png'
        plt.savefig(os.path.join(plots_path, plot_name))
        wandb.log({"recall_by_model_recall_top_" + str(top_k): wandb.Image(os.path.join(plots_path, plot_name))})
        plt.show()

        # Results by f1 score on seen data
        df.sort_values(by=['f1_seen'], ascending=False, inplace=True)
        df.head(top_k).plot(kind='barh', x='checkpoint', y='f1_seen', legend=False, figsize=(15, 10))
        # informative axis labels
        plt.xlabel('F1')
        plt.ylabel('model')
        # show the values on the bars (two decimal places)
        for i, v in enumerate(df.head(top_k)['f1_seen']):
            plt.text(v + 0.001, i, '%.2f' % v, color='black')
        # Add a title
        plt.title('Top ' + str(top_k) + ' models by F1 score on seen data')
        plot_name = 'f1_seen_by_model_f1_seen_top_' + str(top_k) + '.png'
        plt.savefig(os.path.join(plots_path, plot_name))
        wandb.log({"f1_seen_by_model_f1_seen_top_" + str(top_k): wandb.Image(os.path.join(plots_path, plot_name))})
        plt.show()

        # Results by f1 score on unseen data
        df.sort_values(by=['f1_unseen'], ascending=False, inplace=True)
        df.head(top_k).plot(kind='barh', x='checkpoint', y='f1_unseen', legend=False, figsize=(15, 10))
        # informative axis labels
        plt.xlabel('F1')
        plt.ylabel('model')
        # show the values on the bars (two decimal places)
        for i, v in enumerate(df.head(top_k)['f1_unseen']):
            plt.text(v + 0.001, i, '%.2f' % v, color='black')
        # Add a title
        plt.title('Top ' + str(top_k) + ' models by F1 score on unseen data')
        plot_name = 'f1_unseen_by_model_f1_unseen_top_' + str(top_k) + '.png'
        plt.savefig(os.path.join(plots_path, plot_name))
        wandb.log({"f1_unseen_by_model_f1_unseen_top_" + str(top_k): wandb.Image(os.path.join(plots_path, plot_name))})
        plt.show()


if __name__ == '__main__':
    # initialize wandb
    # List of datasets and split to use for evaluation
    # dataset_list = {"args_1": ["/home/tok/figurative-language/datasets/MetaphorRowsDataset.py", "test"],
    #                 "args_2": ["/home/tok/figurative-language/datasets/MetaphorRowsDataset.py", "validation"]}
    # results_path = "/home/tok/figurative-language/figurative-language-data/results_michael"
    # results_path = '/home/tok/figurative-language/results/checkpoints_3'
    # results_path = '/home/tok/figurative-language/results/checkpoints_3'
    # checkpoint_list = [os.path.join(results_path, folder) for folder in os.listdir(results_path)]
    results_path = '/home/tok/figurative-language/figurative-language-data/results_michael'
    checkpoint_list = [

    ]

    # start_index = 10
    # end_index = 30

    name = results_path.split('/')[-1]
    # name = "{}_{}_{}".format(name, start_index, end_index)
    # checkpoint_list = checkpoint_list[start_index:end_index]
    wandb.init(project="fl_results", entity="tokeron", name=name)



    # checkpoint_list = ['/home/tok/figurative-language/results/checkpoints_3/bert_bs_16_ep_5_wl_1.0,10.0,10.0_lr_2e-05_v_base_lc__2022-07-06_09-26-00']
    evaluate_all_models(checkpoint_list, results_path, do_eval=True, plot_results=True)

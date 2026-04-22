import csv


def obtain_label_perm_avg(path, num_task, num_classes):
    """
    obtain labels and acc calculation results from perm_avg csv file
    :param path: path to csv perm_avg file waiting for being read
    :return: labels list, accuracy results
    """
    # list initialization: label list, train accuracy list and test accuracy list
    org_label_list_split, target_label_list_split = [], []  # original labels list, training target label list
    acc_train_avg_list, acc_train_min_list, acc_train_max_list = [], [], []
    acc_test_avg_list, acc_test_min_list, acc_test_max_list = [], [], []

    # labels and acc read from csv file
    n_labels = 2*num_classes*num_task
    with open(path + '.csv', encoding='utf-8-sig') as f:
        for row in csv.reader(f, skipinitialspace=True):
            # orginal labels in classes
            labels = []
            for i in range(num_task):
                labels_per_group = []
                for j in range(num_classes):
                    labels_per_group.append(int(float(row[i*num_classes+j])))
                labels.append(labels_per_group)

            # random target labels in classes
            labels_random = []
            for i in range(num_task):
                labels_per_group_binary_random = []
                for j in range(num_classes):
                    labels_per_group_binary_random.append(int(float(row[num_task*num_classes+i*num_classes+j])))
                labels_random.append(labels_per_group_binary_random)

            org_label_list_split.append(labels)
            target_label_list_split.append(labels_random)

            acc_train_avg_list.append(float(row[n_labels])), acc_train_min_list.append(float(row[n_labels+1])), acc_train_max_list.append(float(row[n_labels+2]))
            acc_test_avg_list.append(float(row[n_labels+3])), acc_test_min_list.append(float(row[n_labels+4])), acc_test_max_list.append(float(row[n_labels+5]))
    f.close()
    return org_label_list_split, target_label_list_split, acc_train_avg_list, acc_train_min_list, acc_train_max_list, acc_test_avg_list, acc_test_min_list, acc_test_max_list


def obtain_label_forget_avg(path, num_task, num_classes):
    """
    obtain labels and forget calculation results from forget_avg csv file
    :param path: path to csv forget_avg file waiting for being read
    :return: labels list, average forgetting results
    """
    # list initialization: label list, average forget performance
    org_label_list_split, target_label_list_split = [], []  # original labels list, training target label list
    acc_forget_avg_list = []  # average forget results collect

    # labels and acc read from csv file
    n_labels = 2*num_classes*num_task
    with open(path + '.csv', encoding='utf-8-sig') as f:
        for row in csv.reader(f, skipinitialspace=True):
            # orginal labels in classes
            labels = []
            for i in range(num_task):
                labels_per_group = []
                for j in range(num_classes):
                    labels_per_group.append(int(float(row[i*num_classes+j])))
                labels.append(labels_per_group)

            # target labels in classes
            labels_random = []
            for i in range(num_task):
                labels_per_group_binary_random = []
                for j in range(num_classes):
                    labels_per_group_binary_random.append(int(float(row[num_task*num_classes+i*num_classes+j])))
                labels_random.append(labels_per_group_binary_random)

            # labels and forget performance obtained
            org_label_list_split.append(labels)
            target_label_list_split.append(labels_random)
            acc_forget_avg_list.append(float(row[n_labels]))

    f.close()
    return org_label_list_split, target_label_list_split, acc_forget_avg_list

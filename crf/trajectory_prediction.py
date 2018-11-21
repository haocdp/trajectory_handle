# ！/usr/bin/env python3
import numpy as np
import sys
from ast import literal_eval
from advanced_tutorial import BiLSTM_CRF
from advanced_tutorial import prepare_sequence
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from collections import Counter


START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4
torch.manual_seed(1)


def validate_trajectory(clusters, regions):
    first_index = 0
    last_index = 0
    new_regions = []
    new_clusters = []
    for i, region in enumerate(regions):
        if i == first_index:
            continue
        if region == regions[first_index] and i != len(regions) - 1:
            last_index = last_index + 1
            continue
        elif i == len(regions) - 1:
            top_one = Counter(clusters[first_index:i + 1]).most_common(1)
            new_regions.append(regions[first_index])
            new_clusters.append(top_one[0][0])
        else:
            top_one = Counter(clusters[first_index:last_index + 1]).most_common(1)
            new_regions.append(regions[first_index])
            new_clusters.append(top_one[0][0])
            first_index = i
            last_index = i
    return new_clusters, new_regions


def main(argv=None):
    if argv is None:
        argv = sys.argv
    filepath = "F:\FCD data\cluster\cluster_region_1.npy"
    trajectory_cluster_region = list(np.load(filepath))

    training_data = []
    test_data = []
    all_count = len(trajectory_cluster_region)
    train_data_length = int(all_count * 0.9)
    count = 0
    labels = []
    for trajectory in trajectory_cluster_region:
        count = count + 1
        clusters = literal_eval(trajectory.strip().split(';')[0])
        regions = literal_eval(trajectory.strip().split(';')[1])

        clusters, regions = validate_trajectory(clusters, regions)
        if len(clusters) < 5 or len(regions) < 5:
            continue

        if count < train_data_length:
            training_data.append((regions, clusters))
        else:
            test_data.append((regions, clusters))
        labels.extend(clusters)

    word_to_ix = {}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    # labels = list(np.load("F:\FCD data\cluster\destination_labels.npy"))
    count = 0
    tag_to_ix = {}
    for label in labels:
        if label not in tag_to_ix.keys():
            tag_to_ix[label] = count
            count = count + 1
    tag_to_ix[START_TAG] = count
    tag_to_ix[STOP_TAG] = count + 1

    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    # print(tag_to_ix)

    # Check predictions before training
    with torch.no_grad():
        precheck_sent = prepare_sequence(test_data[0][0], word_to_ix)
        precheck_tags = torch.tensor([tag_to_ix[t] for t in test_data[0][1]], dtype=torch.long)
        print(model(precheck_sent))

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    count = 0
    all_count = len(training_data)
    for epoch in range(
            1):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence_in, targets)

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()

            count = count + 1
            print(float(count / all_count))

    torch.save(model, "prediction_model.pkl")

    # Check predictions after training
    with torch.no_grad():
        precheck_sent = prepare_sequence(test_data[0][0], word_to_ix)
        print(model(precheck_sent))
    # We got it!


if __name__ == '__main__':
    sys.exit(main())
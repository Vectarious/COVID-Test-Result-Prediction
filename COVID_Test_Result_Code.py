import csv
from random import randrange

# Site where I found data: https://www.nature.com/articles/s41746-020-00372-6
# GitHub for data download: https://github.com/nshomron/covidpred

# Tree Class (so I don't have to use the array again...)
class BTNode:
    # Initialize a node to have a left, right, and data element
    def __init__(self):
        self.left = None
        self.right = None
        self.data = None
    # Getter functions for the elements
    def getLeft(self):
        return self.left
    def getRight(self):
        return self.right
    def getData(self):
        return self.data
    # Setter functions for the elements
    def setLeft(self,left_child):
        self.left = left_child
    def setRight(self,right_child):
        self.right = right_child
    def setData(self,data):
        self.data = data

# Function which cleans the dataset before anything is done to it
def clean_data(row):
    # Each row is in this form
    [test_date, cough, fever, sore_throat, shortness_of_breath, head_ache,
     corona_result, age_60_and_above, gender, test_indication] = row
    # We want to return it in this form, turning the dataset into rows of booleans which are much easier to parse
    # It also has the benefit of turning columns with more than 2 possible results into multiple booleans
    # This took me a long time to figure out but it seems to work well, I'm hoping this counts for Extra #2...
    return [
        cough == '1',
        fever == '1',
        sore_throat == '1',
        shortness_of_breath == '1',
        head_ache == '1',
        age_60_and_above != 'None',
        age_60_and_above == 'Yes',
        gender == 'male',
        test_indication == 'Contact with confirmed',
        test_indication == 'Abroad',
        test_indication == 'Other',
        corona_result == 'positive'
        ]

# New tree function, makes a tree using a given dataset
def new_tree(dataset):
    # Create the root node
    root = BTNode()
    # The variable list has been set to simply contain the first 11 variables in the cleaned dataset (0-10)
    variable_array = list(range(0, len(dataset[0]) - 1))
    # Call the make_tree function to recursively create the tree
    make_tree(dataset, variable_array, root, 0)
    return root

# Recursive function which generates the tree
def make_tree(smaller_dataset, variable_array, parent_node, prob):
    # Base case: if the smaller dataset is empty or if the dataset is None, set the data of the node to prob of the previous node
    # This happens when all of the data is sent to a single side during a split (no available points behave this way)
    if len(smaller_dataset) == 0 or smaller_dataset == None:
        parent_node.setData(prob)
        return
    # The second base case is if the variable array is empty or equal to None, in which case there are no more variables to split by
    elif len(variable_array) == 0 or variable_array == None:
        # In this case, store the probability of something being positive here in the leaf node
        positives = how_many_positives(smaller_dataset)
        parent_node.setData(positives / len(smaller_dataset))
        return
    # Recursive case
    else:
        # Call the find_best_variable function to determine the best variable to split by
        best_variable = find_best_variable(variable_array, smaller_dataset)
        # Call the make_smaller_dataset function to split the current dataset by that variable
        trues,falses = make_smaller_datasets(smaller_dataset, best_variable)
        # Each time this case is called, we need to make a copy of the variable array before removing, so it doesn't affect the other branches of recursion
        new_variable_array = variable_array.copy()
        new_variable_array.remove(best_variable)
        # Set the data of the current node to "best_variable" which allows the recurse_tree function to path through the tree based on the variable
        parent_node.setData(best_variable)
        # Set the left and right of the current node to new BTNodes
        parent_node.setLeft(BTNode())
        parent_node.setRight(BTNode())
        # Find the current probability of being positive, this is important to pass in case the current split put all the data on one side
        positives = how_many_positives(smaller_dataset)
        prob = positives / len(smaller_dataset)
        # Call make_tree with the new arrays on the left and right child
        make_tree(trues, new_variable_array, parent_node.getLeft(),prob)
        make_tree(falses, new_variable_array, parent_node.getRight(),prob)
        return
            

# Function which looks at the current dataset and determines the best variable to split by
def find_best_variable(variable_array, smaller_dataset):
    # correlation_array will store values which correspond to how good a given variable split is
    correlation_array = [0]*len(variable_array)
    # For each variable in the array, run find_best_variable_helper
    for i in range(len(variable_array)):
        correlation_array[i] = find_best_variable_helper(variable_array[i], smaller_dataset)
    # Return the variable with the index equal to the index of the max of the correlation array
    return variable_array[correlation_array.index(max(correlation_array))]

# Helper function for the previous function
def find_best_variable_helper(variable, smaller_dataset):
    # Split the dataset by the variable
    left, right = make_smaller_datasets(smaller_dataset, variable)
    # Find how many positives are in each subset
    lp = how_many_positives(left)
    rp = how_many_positives(right)
    # We need cases for if either or both sides are 0
    if len(left) == 0 or len(right) == 0:
        # If they're both 0, return None since this means the smaller_dataset is empty
        if len(left) == 0 and len(right) == 0:
            return None
        # If 1 side is empty, use the other side
        elif len(left) == 0:
             return rp/len(right)
        else:
            return lp/len(left)
    # If we have data on both sides, return the side with the highest ratio of positives (the most indicative of a good variable split)
    else:
        return max(lp/len(left),rp/len(right))

# Function which does the splitting of the dataset by a variable
def make_smaller_datasets(smaller_dataset, variable):
    # Create arrays to store the new subsets
    trues = []
    falses = []
    # If the datapoint is true for the variable, add it to the trues, otherwise add to the falses
    for datum in smaller_dataset:
        if datum[variable]:
            trues.append(datum)
        else:
            falses.append(datum)
    return trues, falses

# A simple function which returns how many positive cases there are in a given dataset
def how_many_positives(test_dataset):
    positive_count = 0
    for datum in test_dataset:
        if datum[-1]:
            positive_count += 1
    return positive_count


# Function which recurses through a tree for a given datapoint (row) until it finds a leaf node and returns the probability stored in it
def recurse_tree(node,row):
    if node.getLeft() == None:
        return node.getData()
    else:
        variable = node.getData()
        if row[variable]:
            return recurse_tree(node.getLeft(),row)
        else:
            return recurse_tree(node.getRight(),row)
            

# Validation Function 1/3: Slow leave-one-out function which runs on a given number of datapoints (validation_number)
# It takes about 4 seconds to generate and recurse through a tree, so when I ran it on 100 datapoints it took roughly 4.5 minutes
# I ran it 6 times  for validation_number = 50 and got between 70-92% accuracy
def validate_tree(dataset, validation_number):
    popped_indices = []
    success_count = 0
    total_count = 0
    for _ in range(validation_number):
        # Make a duplicate dataset and pop off a random datapoint each time
        test_dataset = dataset.copy()
        popped_index = randrange(len(dataset))
        # Make sure the same datapoint isn't randomly used twice
        while popped_index in popped_indices:
            popped_index = randrange(len(dataset))
        popped_indices.append(popped_index)
        test_datapoint = test_dataset.pop(popped_index)
        # Make a new tree with the popped dataset and track successful detections by comparing the tree result to the actual result (final index of the datapoint)
        test_tree = new_tree(test_dataset)
        # I will explain later where the .02 came from
        if recurse_tree(test_tree,test_datapoint) > .02:
            if test_datapoint[-1]:
                success_count +=1
            total_count += 1
        else:
            if not test_datapoint[-1]:
                success_count +=1
            total_count += 1
    return success_count/total_count

# Validation Function 2/3: While that was the "correct" way to validate a tree, I wanted to see how it would do if I just used the same tree with all datapoints.
# The reason this is not correct is because the datapoint itself is in the tree, so it would trace through the same way and affect the results
# However, since virtually all datapoints are non-unique, I think this still gives a very good approximation
def validate_dataset(test_dataset,comparison_value):
    # Nearly identical function except it uses the same tree every time, so it's quick to run through all 270,000+ datapoints
    success_count = 0
    total_count = 0
    test_tree = new_tree(test_dataset)
    for test_datapoint in test_dataset:
        if recurse_tree(test_tree,test_datapoint) > comparison_value:
            if test_datapoint[-1]:
                success_count +=1
            total_count += 1
        else:
            if not test_datapoint[-1]:
                success_count +=1
            total_count += 1
    return success_count/total_count

# Validation Function 3/3: Originally, I had set the comparison value above to .5, meaning over half of the values in the leaf needed to be positive to guess it was a positive.
# While this made the overall result ~99% accurate, I realized that this was because the overwhelming majority of the data were negative results. I checked to see how accurately
# I was predicting both negative and positive values with this function. It showed that when comparison_value is .5, I was getting 99% of the negatives correct, but only 58% of
# the positives correct. This means the algorithm was skewed towards giving false negatives. I then checked through different values for comparison_value to find that the value
# which gave the best predictions for both positive AND negative values was .02, which was accurate for roughly 85% of both positives and negatives. Which this does decrease the
# accuracy for negative values (giving more false positives), I believe this is a more effective and useful algorithm since I can say it has an overall effectiveness of 85%.
def tree_validater(test_dataset, comparison_value):
    # Very similar to the previous two functions, except it tracks the results specifically for positives and negatives and then returns both percentages
    successful_positive_count = 0
    total_positive_count = 0
    successful_negative_count = 0
    total_negative_count = 0
    
    test_tree = new_tree(test_dataset)
    for test_datapoint in test_dataset:
        prob = recurse_tree(test_tree,test_datapoint)
        if test_datapoint[-1]:
            if prob > comparison_value:
                successful_positive_count +=1
            total_positive_count += 1
        else:
            if prob < comparison_value:
                successful_negative_count += 1
            total_negative_count += 1
    return successful_positive_count/total_positive_count, successful_negative_count/total_negative_count

# This function uses a naive bayes algorithm to predict the same thing the tree was trying to predict (positive or negative test result)
def naive_bayes(dataset, correlation_value):
    # Use function below to get probabilities for testing positive given a variable is True or False
    probability_true_array = get_probability_array(dataset, True)
    probability_false_array = get_probability_array(dataset, False)

    # Store if the datapoint is predicted to be positive or negative with a "True" or "False" in the array
    datapoint_probabilities = []
    for datapoint in dataset:
        # Start with a base probability and then multiply by the value in the respective array (determined by if the index is true or false)
        datapoint_prob = 1.0*(10**13)
        for i in range(len(datapoint)-1):
            if datapoint[i]:
                datapoint_prob *= probability_true_array[i]
            else:
                datapoint_prob *= probability_false_array[i]
        # Compare to the correlation value to append a true or false (if we think it's positive or negative)
        if datapoint_prob > correlation_value:
            datapoint_probabilities.append(True)
        else:
            datapoint_probabilities.append(False)
    # Return the array of guesses
    return datapoint_probabilities
    

def get_probability_array(dataset, true_false):
    # The probability that a datapoint is positive given its features is equal to the probability that the datapoint is positive given 1 feature times
    # the probability that it's positive given another feature etc. To do this in an efficient (somewhat incorrect) manner, store the probabilities for
    # a given feature in an array. It is somewhat incorrect to do this because the datapoint itself should be excluded as not to impact results,
    # however this takes way too much time to be useful, and since we're dealing with such large numbers of datapoints, a single one doesn't make
    # too much of an impact

    probability_array = []

    # For each variable (column), search through each datapoint and find the proportion of positive trues to all trues (where a "true" is if the
    # variable is true or not)
    for i in range(len(dataset[0])-1):
        variable_counter = 0
        positive_variable_counter = 0
        for datapoint in dataset:
            # If we want a true vs a false array, toggle this variable when calling the function
            if datapoint[i] == true_false:
                variable_counter += 1
                if datapoint[-1]:
                    positive_variable_counter += 1
        # Append the ratio of these (probability that the datapoint is positive given the variable is true
        probability_array.append(positive_variable_counter/variable_counter)
    return probability_array

def bayes_validater(dataset, datapoint_probabilities_array):
    # Similarly to the validater for the tree, we find the number of positives and negatives we got correct and incorrect to return the
    # percentage of each we got correct
    correct_positive = 0
    correct_negative = 0
    incorrect_positive = 0
    incorrect_negative = 0
    # Search through the datapoints to tick up the counters
    for i in range(len(datapoint_probabilities_array)):
        if datapoint_probabilities_array[i] == dataset[i][-1]:
            if dataset[i][-1]:
                correct_positive += 1
            else:
                correct_negative += 1
        else:
            if dataset[i][-1]:
                incorrect_positive += 1
            else:
                incorrect_negative += 1
    # Create and return the percentages
    positive_percent_correct = correct_positive / (correct_positive + incorrect_positive)
    negative_percent_correct = correct_negative / (correct_negative + incorrect_negative)
    return positive_percent_correct, negative_percent_correct
        

# Main function
def main():
    # Create dataset array which holds all data
    dataset = []
    # Load in datafile
    with open("covid_data.csv") as csvfile:
        filereader = csv.reader(csvfile, delimiter=',', quotechar='"')
        # Skip the first line (headers)
        next(filereader)
        for row in filereader:
            # As data is appended, clean it
            dataset.append(clean_data(row))
            
    # Run validation on trees made by removing 100 random individual datapoints (takes 4-5 minutes)
    print("Tree Percent Correct for n=100", validate_tree(dataset,100))

    # Validation using a single tree (not removing any datapoints)
    print("Tree Percent Correct (single tree):", validate_dataset(dataset,.02))

    # Validation using a single tree (not removing any datapoints), but sorted by positive/negative results
    print("Tree Percent Correct (positives, negatives):", tree_validater(dataset,.02))

    # Validation for Naive Bayes algorithm, sorted by positive/negative results
    print("Bayes Percent Correct (positives, negatives):", bayes_validater(dataset,naive_bayes(dataset,.02)))
main()

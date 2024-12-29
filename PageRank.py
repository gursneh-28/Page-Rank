import pandas as pd
import numpy as np

def page_rank(matrix, d=0.85, max_iterations=100, tolerance=1.0e-6):
    # Step 1: Initialize PageRank values
    n = matrix.shape[0]  # Number of students/nodes
    page_rank = np.ones(n) / n  # Initial PageRank: 1/n for each node

    # Step 2: Power iteration
    for iteration in range(max_iterations):
        new_page_rank = np.zeros(n)  # Initialize new PageRank values
        for i in range(n):
            # Calculate the rank for each node (student)
            incoming_links = matrix.iloc[:, i]  # People who are impressed by person i
            new_page_rank[i] = (1 - d) / n + d * np.dot(page_rank, incoming_links)
        
        # Step 3: Check for convergence
        if np.linalg.norm(new_page_rank - page_rank, 1) < tolerance:
            print(f"Converged after {iteration + 1} iterations.")
            break
        
        page_rank = new_page_rank  # Update PageRank for the next iteration

    return page_rank

# Load the CSV file (replace with the correct path)
file_path = 'data - data.csv'
data = pd.read_csv(file_path, index_col=0)  # Use index_col=0 to set the first column as the index

# Initialize a set to hold all unique impressive people
all_students = set()

# Loop through each column to collect unique impressive people
for column in data.columns:
    unique_students = set(data[column].dropna().unique())  # Get unique impressive people for each column
    all_students.update(unique_students)  # Add to the set of all impressive people

# Convert the set to a sorted list for consistent output
all_students = sorted(all_students)
n = len(all_students)

matrix = pd.DataFrame(np.zeros((n, n)),index=all_students, columns=all_students)

# Populate the adjacency matrix
for person_a in data.index:
    impressed_people = data.loc[person_a].dropna().values  
    for person_b in impressed_people:
        matrix.loc[person_a, person_b] = 1  

for i in range(0,n):
    outgoing_links =np.sum(matrix.iloc[i,:])  
    if outgoing_links>0:
        matrix.iloc[i, :] = matrix.iloc[i, :] / outgoing_links
    else:
        matrix.iloc[i,:] = 1/n

# Run PageRank algorithm on the matrix
page_rank_values = page_rank(matrix)

# Add PageRank values back to a DataFrame for better visualization
page_rank_df = pd.DataFrame({'Student': matrix.columns, 'PageRank': page_rank_values})

# Sort the students by PageRank in descending order
top_students = page_rank_df.sort_values(by='PageRank', ascending=False)

top_students = top_students.reset_index(drop=True)

# Print the top 10 most popular students
print("Top 10 most popular students:")
print(top_students.head(10))
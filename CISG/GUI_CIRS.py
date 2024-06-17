import time
import tkinter as tk
from tkinter import ttk, messagebox, Text, END
from PIL import Image, ImageTk
from SearchFiles import search_plots, tfidf_vectorizer, tfidf_matrix

# GUI Event handler for SEARCH BUTTON widget
def on_search():
    query = search_entry.get()  # Retrieves the text entered
    if query:  # = ! an empty string
        start_time = time.time()  # Start time measuring
        sorted_titles, sorted_plots, sorted_similarities = search_plots(query, tfidf_vectorizer, tfidf_matrix)
        end_time = time.time()  # End time measuring
        elapsed_time = end_time - start_time  # Search time
        display_results(sorted_titles, sorted_plots, sorted_similarities, elapsed_time)
    else:
        messagebox.showwarning("Input required", "Please enter a search query")

# Function to limit plot text in main field to 70 characters
def truncate_plot(plot):
    if len(plot) > 70:
        return plot[:70] + "... [click for more details]"
    else:
        return plot + " [= Full Plot]"

# Function to display results in the GUI
def display_results(titles, plots, similarities, elapsed_time):
    # Clear the existing results, item = row
    for item in treeview_result_list.get_children():
        treeview_result_list.delete(item)
    # Clear the full_plot text box
    full_plot.delete('1.0', END)
    # Filter results with similarity > 0.1, zip() combines lists into list of tuples
    filtered_results = [(title, plot, similarity) for title, plot, similarity in zip(titles, plots, similarities) if similarity > 0.1]
    if filtered_results:
        for title, plot, similarity in filtered_results[:10]:  # Max. 10 rows
            short_plot = truncate_plot(plot)
            # Inserts new row into Treeview widget (title, similarity, plot)
            treeview_result_list.insert('', 'end', values=(title, f"{similarity:.4f}", short_plot, plot))
    else:
        treeview_result_list.insert('', 'end', values=("<no significant similarities>", "<no significant similarities>", "<no significant similarities>"))
        full_plot.insert(END, "<no significant similarities>\n")
    # Display calculated time
    time_label.config(text=f"Search time taken: {elapsed_time:.4f} seconds")

# Function (event = <<TreeviewSelect>>):
# - 1. Retrieve selected item from Treeview widget
# - 2. Checks if item selected
# - 3. Gets details of selected item
# - 4. Extracts plot information of selected row
# - 5. Updates full_plot (clears current and inserts new plot info)
def on_row_select(event):
    selected_item = treeview_result_list.selection()  # Returns tuple of selected items in Treeview widget
    if selected_item:  # Checks if item selected (true)
        item = treeview_result_list.item(selected_item)  # Retrieves details
        plot = item['values'][3]  # Choosing full plot info in third column
        full_plot.delete('1.0', END)  # Deletes current plot content
        full_plot.insert(END, plot)  # Inserts new plot content into widget

# APPLICATION WINDOW
root = tk.Tk()
root.title("CIRS - Movie Plot Search App")
root.geometry("1000x700")
root.resizable(False, True)  # fixed width, resizable length

# Configure the grid to expand and contract properly
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_rowconfigure(2, weight=1)
root.grid_rowconfigure(3, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure(2, weight=1)
root.grid_columnconfigure(3, weight=1)

# CIRS PICTURE
image = Image.open("CIRS.png").resize((100, 100))
photo = ImageTk.PhotoImage(image)

# Style definition for widgets
style_search_entry = ttk.Style()
style_search_entry.configure("TEntry", padding=5, borderwidth=5, relief="solid")

# LABEL search widget
search_label = ttk.Label(root)
search_label.grid(row=0, column=0, padx=10, pady=0)
search_label.config(image=photo)

# INPUT widget
search_entry = ttk.Entry(root, width=50)
search_entry.grid(row=0, column=1, padx=10, pady=5)
search_entry.config(font=("Arial", 15), style="TEntry")

# SEARCH BUTTON widget
search_button = ttk.Button(root)
search_button.grid(row=0, column=2, padx=0, pady=5, sticky="w")
search_button.config(text="Search", command=on_search)

# TREEVIEW widget (= multi-column list or table)
columns = ('Title', 'Similarity', 'Short Plot', 'Full Plot')
treeview_result_list = ttk.Treeview(root, columns=columns, show='headings', height=10)
for col in columns:
    treeview_result_list.heading(col, text=col)
treeview_result_list.column('Title', width=100)
treeview_result_list.column('Similarity', width=50, anchor="center")
treeview_result_list.column('Short Plot', width=500)
treeview_result_list.column('Full Plot', width=0, stretch=tk.NO)  # Hidden column for full plot
treeview_result_list.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky='nsew')

# <<TreeviewSelect>> = virtual event to detect when a selection in a Treeview widget changes
treeview_result_list.bind('<<TreeviewSelect>>', on_row_select)

# LABEL widget for plot
plot_label = ttk.Label(root, text="Full Plot")
plot_label.grid(row=2, column=1, padx=10, pady=5)
plot_label.config(font=("Arial", 10))

# TEXT widget for full_plots
full_plot = Text(root, wrap='word', height=10)
full_plot.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky='nsew')

# LABEL widget for search time
time_label = ttk.Label(root, text="")
time_label.grid(row=4, column=0, columnspan=3, padx=10, pady=5)

# Run the application
root.mainloop()

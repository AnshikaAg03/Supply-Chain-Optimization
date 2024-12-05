from flask import Flask, request, render_template, redirect, url_for
from flask_cors import CORS  # Import Flask-CORS
import os
import pandas as pd
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, value, LpStatus
import pulp
import json
import plotly.graph_objects as go
from plotly.offline import plot
import numpy as np
import subprocess
import plotly.io as pio
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib
import logging
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
matplotlib.use('Agg')


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes by default
logging.basicConfig(level=logging.DEBUG)

UPLOAD_FOLDER = 'uploads/'  # Ensure this folder exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Store uploaded files data for use across routes
uploaded_data = {}
uploaded_new_data={}

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'xlsx', 'xls'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/home')
def home():
    return render_template('upload.html',uploaded_data='',show_home=True)
@app.route('/')
def upload_form():
    return render_template('upload.html', uploaded_data='', show_upload_form=True)
@app.route('/upload', methods=['POST'])
def upload_files():
    global uploaded_data  # Use the global variable to store uploaded data

    if 'files' not in request.files:
        return 'No file part'
    
    files = request.files.getlist('files')

    if not files or all(file.filename == '' for file in files):
        return 'No selected files'
    
    uploaded_files_info = []

    for file in files:
        if file and allowed_file(file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            data = pd.read_excel(file_path)

            # Set the first column as the index
            data.set_index(data.columns[0], inplace=True)

            # Store the DataFrame in the global dict
            uploaded_data[file.filename] = data  
            uploaded_files_info.append(f'File: {file.filename}<br>Data:<br>{data.head().to_html()}')
        else:
            uploaded_files_info.append(f'Invalid file type for {file.filename}')
    
    return render_template('upload.html', uploaded_data='<br><br>'.join(uploaded_files_info), show_upload_form=True)

@app.route('/preview')
def preview_files(): 
    global uploaded_data
    preview_info = ''  # String to store the combined HTML preview
    
    for filename, data in uploaded_data.items():
        # Reset the index to include it in the HTML output
        data_preview = data.reset_index().head().to_html(classes='table table-striped table-bordered', index=False)

        # Combine everything into preview_info
        preview_info += f'''
            <div class="mb-4">
                <h2 class="text-danger">{filename}</h2>
                <div class="table-responsive">
                    {data_preview}
                </div>
            </div>
        '''
    
    # Render the HTML using render_template, passing the preview_info
    return render_template('upload.html', uploaded_data=None, preview_info=preview_info, show_upload_form=False)


@app.route('/summarize')
def summarize_data():
    global uploaded_data
    global fixed_costs,demand,var_cost,cap,freight_costs
    summary_info = ''
    filenames = list(uploaded_data.keys())

    for index, filename in enumerate(filenames):
        if index==0 :
            cap = uploaded_data[filename]
        elif index==1:
            demand=uploaded_data[filename]
        elif index==2:
            fixed_costs=uploaded_data[filename]
        elif index==3:
            freight_costs=uploaded_data[filename]
        elif index==4:
            var_cost=uploaded_data[filename] 
    status_out, objective_out, y, x, fix, var = optimization_model(fixed_costs, var_cost, demand, demand.columns[0], cap)
    summary_info={
        'status': status_out,
        'objective_value': objective_out,
        'fixed_costs': fix,
        'variable_costs': var,
    }
# Example factory locations with coordinates
    location_coords = {
    'USA': [-95.7129, 37.0902],
    'GERMANY': [10.4515, 51.1657],
    'JAPAN': [138.2529, 36.2048],
    'BRAZIL': [-51.9253, -14.2350],
    'INDIA': [78.9629, 20.5937]
}

# Create a list of all connections between locations
    connections = [(loc1, loc2) for loc1 in location_coords for loc2 in location_coords if loc1 != loc2]

# Create an empty figure
    fig = go.Figure() 

# Add markers for factory locations
    for loc, coord in location_coords.items():
        if y[(loc, 'LOW')].varValue == 0 and y[(loc, 'HIGH')].varValue == 0:
            marker_color = 'red'  # Closed
        elif y[(loc, 'LOW')].varValue == 1 and y[(loc, 'HIGH')].varValue == 0:
            marker_color = 'blue'  # Open with low capacity
        elif y[(loc, 'LOW')].varValue == 0 and y[(loc, 'HIGH')].varValue == 1:
            marker_color = 'green'  # Open with high capacity
        elif y[(loc, 'LOW')].varValue == 1 and y[(loc, 'HIGH')].varValue == 1:
            marker_color = 'purple'  # Both capacities open
        fig.add_trace(go.Scattergeo(
            lon=[coord[0]], lat=[coord[1]],
            text=loc,
            hovertext = f'{loc}<br>Demand => {demand.loc[loc, "Demand"]}<br>Capacity = {cap.loc[loc, "LOW"]}(L) , {cap.loc[loc, "HIGH"]}(H) ',
            mode='markers+text',
            marker=dict(size=10, color=marker_color),
            name=f'{loc} Factory' # Legend entry for each factory 
        )) 

    # Add curved lines (connections) between the factory locations
    for idx, connection in enumerate(connections):
        start, end = connection
        lon_start, lat_start = location_coords[start]
        lon_end, lat_end = location_coords[end] 

        # Create curvature for each line by introducing intermediate points
        lon_curve = np.linspace(lon_start, lon_end, 500) 
        lat_curve = np.linspace(lat_start, lat_end, 500) + np.sin(np.linspace(0, np.pi, 500)) *np.random.uniform(1,20)  # Adding curve 

        if y[(start, 'LOW')].varValue == 0 and y[(start, 'HIGH')].varValue == 0:
            # Factory is closed, use red color for the line
            line_color = 'red'
        else:
            # Factory is open, use a unique color
            line_color = f'rgba({idx*40%255}, {100+idx*20%255}, {idx*60%255}, 0.7)'

        # Add line traces with hover information and interactive legend
        fig.add_trace(go.Scattergeo(
            lon=lon_curve,
            lat=lat_curve,
            mode='lines',
            line=dict(width=2, color=line_color),  # Unique color
            hoverinfo='text',  # Enable hover info
            text=f'{start} to {end} - Info about this path',  # Hover text
            name=f'{start} to {end}'  # Line name in the legend
        )) 

    fig.update_layout(
        annotations=[
            dict(
                x=0, y=0.85,
                xref='paper', yref='paper',
                text='Factory Status:',  # Main title for the custom legend
                showarrow=False,
                font=dict(size=14, color="black"),
            ),
            dict(
                x=0, y=0.75,
                xref='paper', yref='paper',
                text='<span style="color:red;">●</span> Closed Factory',
                showarrow=False,
                font=dict(size=12),
            ),
            dict(
                x=0, y=0.65,
                xref='paper', yref='paper',
                text='<span style="color:blue;">●</span> Low Capacity Factory',
                showarrow=False,
                font=dict(size=12),
            ),
            dict(
                x=0, y=0.57,
                xref='paper', yref='paper',
                text='<span style="color:green;">●</span> High Capacity Factory',
                showarrow=False,
                font=dict(size=12),
            ),
            dict(
                x=0, y=0.50,
                xref='paper', yref='paper',
                text='<span style="color:purple;">●</span> Both Capacities Factory',
                showarrow=False,
                font=dict(size=12),
            ),
        ]
    )

    # Update layout to include the interactive legend
    fig.update_layout(
        paper_bgcolor='lightyellow',  # Color of the entire paper
        plot_bgcolor='lightyellow',
        title="Factory Locations and Interactive Connections",
        showlegend=True,  # Display legend
        geo=dict(
            projection_type="orthographic", 
            showcoastlines=True, 
            coastlinecolor="black", 
            landcolor="lightgreen", 
            showocean=True, 
            oceancolor="lightblue", 
            showcountries=True, 
            countrycolor='black',
            resolution=110,
        ),
        legend=dict(  # Customizing the legend
            title="Factory Connections",
            traceorder="normal",  # Keep order of traces
            font=dict(size=10),
            bgcolor="LightSteelBlue",
            bordercolor="Black",
            borderwidth=2
        )
    )
    plot_html = pio.to_html(fig, full_html=False)
    # Render the HTML with summary and optimization results
    return render_template('upload.html',
                           summary_info=summary_info, 
                           show_upload_form=False,
                           plot_html=plot_html, 
                           show_summarization=True,  # This ensures the summary tab is shown
                           show_preview=False)

def optimization_model(fixed_costs, var_cost, demand, demand_col, cap):
    '''Build the optimization based on input parameters'''
    # Define Decision Variables
    global loc,size
    loc = ['USA', 'GERMANY', 'JAPAN', 'BRAZIL', 'INDIA']
    size = ['LOW', 'HIGH']
    global plant_name 
    plant_name= [(i,s) for s in size for i in loc]
    prod_name = [(i,j) for i in loc for j in loc]   

    # Initialize Class
    model = LpProblem("Capacitated Plant Location Model", LpMinimize)

    # Create Decision Variables
    x = LpVariable.dicts("production_", prod_name,
                         lowBound=0, upBound=None, cat='continuous')
    y = LpVariable.dicts("plant_", 
                         plant_name, cat='Binary')

    # Define Objective Function
    model += (lpSum([fixed_costs.loc[i,s] * y[(i,s)] * 1000 for s in size for i in loc])
              + lpSum([var_cost.loc[i,j] * x[(i,j)]   for i in loc for j in loc]))

    # Add Constraints
    for j in loc:
        model += lpSum([x[(i, j)] for i in loc]) == demand.loc[j,demand_col]
    for i in loc:
        model += lpSum([x[(i, j)] for j in loc]) <= lpSum([cap.loc[i,s]*y[(i,s)] * 1000
                                                           for s in size])                                                 
    # Solve Model
    model.solve()
    
    # Results
    status_out = LpStatus[model.status]
    objective_out  = pulp.value(model.objective)
    plant_bool = [y[plant_name[i]].varValue for i in range(len(plant_name))]
    fix = sum([fixed_costs.loc[i,s] * y[(i,s)].varValue * 1000 for s in size for i in loc])
    var = sum([var_cost.loc[i,j] * x[(i,j)].varValue for i in loc for j in loc])
    plant_prod = [x[prod_name[i]].varValue for i in range(len(prod_name))]
    return status_out, objective_out, y, x, fix, var

def create_unique_combination_plot(df_unique):
    fig, ax = plt.subplots(figsize=(12, 4))

    # Set the figure background color
    fig.patch.set_facecolor('lightyellow')  # Set the background color of the entire figure

    # Set the axes background color
    ax.set_facecolor('lightyellow')  # Set the background color of the axes

    # Create the pcolor plot
    c = ax.pcolor(df_unique, cmap='Blues', edgecolors='k', linewidths=0.5)

    ax.set_xticks([i + 0.5 for i in range(df_unique.shape[1])])
    ax.set_xticklabels(df_unique.columns, rotation=90, fontsize=12)
    ax.set_yticks([i + 0.5 for i in range(df_unique.shape[0])])
    ax.set_yticklabels(df_unique.index, fontsize=12)

    plt.tight_layout()

    # Save plot to a BytesIO object to embed later
    img = io.BytesIO()
    FigureCanvas(fig).print_png(img)
    plt.close(fig)  # Close the figure to avoid display in non-web environments

    # Encode to base64 so it can be embedded in HTML
    img_base64 = base64.b64encode(img.getvalue()).decode()
    return img_base64

@app.route('/demand_simulation', methods=['GET', 'POST'])
def demand_simulation():
    # Demand simulation logic (same as before)
    N = 50
    df_demand = pd.DataFrame({'scenario': np.array(range(1, N + 1))})
    data = demand.reset_index()  # Assuming demand is defined
    CV = 0.5
    markets = data['(Units/month)'].values

    for col, value in zip(markets, data['Demand'].values):
        sigma = CV * value
        # Generate normally distributed values
        df_demand[col] = np.random.normal(value, sigma, N)

        # Ensure that demand values are at least 1
        df_demand[col] = df_demand[col].apply(lambda t: max(t, 1))

    # Add Initial Scenario
    COLS = ['scenario'] + list(demand.index)
    VALS = [0] + list(demand['Demand'].values)
    df_init = pd.DataFrame(dict(zip(COLS, VALS)), index=[0])

    # Concat
    df_demand = pd.concat([df_init, df_demand])

    # Save to Excel
    excel_path = f"C:\\Users\\Hp-D\\OneDrive\\Desktop\\supplyapp\\flask-server\\api\\uploads\\df_demand-{int(CV * 100)}PC.xlsx"
    df_demand.to_excel(excel_path)


    # Create interactive plot using Plotly
    traces = []
    colors = ['green', 'red', 'black', 'blue', 'orange']
    
    for i, col in enumerate(markets):
        trace = go.Scatter(
            x=df_demand['scenario'],
            y=df_demand[col],
            mode='lines+markers',
            name=col,
            line=dict(color=colors[i % len(colors)]),
            marker=dict(size=8)
        )
        traces.append(trace)

    # Create layout for the plot
    layout = go.Layout(
        title='Demand Simulation Results',
        xaxis=dict(title='Scenario'),
        yaxis=dict(title='(Units)'),
        showlegend=True
    )
    # Create a figure
    fig = go.Figure(data=traces, layout=layout)

    uploads_dir = os.path.join('static', 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)

    # Now save the plot
    plot_path = os.path.join(uploads_dir, 'demand_simulation_plot.html')
    fig.update_yaxes(type='log')
    plot(fig, filename=plot_path, auto_open=False)

    # Convert to HTML for rendering
    demand_simulation_results = df_demand.head().astype(int).to_html(classes='table table-striped', index=False)

    # Record results per scenario
    list_scenario, list_status, list_results, list_totald, list_fixcost, list_varcost = [], [], [], [], [], []
    # Initial Scenario
    status_out, objective_out, y, x, fix, var = optimization_model(fixed_costs, var_cost, demand, 'Demand', cap)

    # Add results
    list_scenario.append('INITIAL')
    total_demand = demand['Demand'].sum()
    list_totald.append(total_demand)
    list_status.append(status_out)
    list_results.append(objective_out)
    list_fixcost.append(fix)
    list_varcost.append(var)
    # Dataframe to record the solutions
    df_bool = pd.DataFrame(data = [y[plant_name[i]].varValue for i in range(len(plant_name))], index = [i + '-' + s for s in size for i in loc], 
                            columns = ['INITIAL'])
    
    # Simulate all scenarios
    demand_var = df_demand.drop(['scenario'], axis = 1).T  
    # Loop
    for i in range(1, 50): # 0 is the initial scenario 
        # Calculations
        status_out, objective_out, y, x, fix, var = optimization_model(fixed_costs, var_cost, demand_var, i, cap) 
        # Append results
        list_status.append(status_out)
        list_results.append(objective_out)
        df_bool[i] = [y[plant_name[i]].varValue for i in range(len(plant_name))]
        list_fixcost.append(fix)
        list_varcost.append(var)
        total_demand = demand_var[i].sum()
        list_totald.append(total_demand)
        list_scenario.append(i)
    # Final Results
    # Boolean
    df_bool = df_bool.astype(int)
    df_bool.to_excel(f'C:\\Users\\Hp-D\\OneDrive\\Desktop\\supplyapp\\flask-server\\api\\uploads\\boolean-{int(CV * 100)}PC.xlsx')

        # Create heatmap with grid-like effect
    fig = go.Figure(data=go.Heatmap(
        z=df_bool.values,
        x=df_bool.columns,
        y=df_bool.index,
        colorscale='Blues',
        zmin=0,
        zmax=1,
        colorbar=dict(title="Value"),
        hoverinfo='z'  # Show only the value on hover
    ))

    # Update layout to emphasize gridlines
    fig.update_layout(
        paper_bgcolor='lightyellow',
        plot_bgcolor='lightyellow',
        title='Demand Simulation - Boolean Heatmap',
        xaxis=dict(
            showgrid=True,  # Show vertical grid lines 
            dtick=1,  # Set tick distance (so every column has a tick)
            tickvals=list(df_bool.columns),  # Show all columns as ticks
            ticks='outside',
            showline=True,  # Draw the x-axis line
            linewidth=2,  # Thicker x-axis line
            linecolor='black'  # Set color of the line
        ),
        yaxis=dict(
            showgrid=True,  # Show horizontal grid lines
            dtick=1,  # Set tick distance (so every row has a tick)
            tickvals=list(df_bool.index),  # Show all rows as ticks
            ticks='outside',
            showline=True,  # Draw the y-axis line
            linewidth=2,  # Thicker y-axis line
            linecolor='black'  # Set color of the line
        ),
        height=600,
        width=1200,
        xaxis_nticks=50,
    )

    # Ensure a grid-like definition for cells
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', gridcolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', gridcolor='black')

    # Save or render the heatmap
    plot_html_2 = fig.to_html(full_html=False)

    df_unique = df_bool.T.drop_duplicates().T
    df_unique.columns = ['INITIAL'] + ['C' + str(i) for i in range(1, len(df_unique.columns))]

    # Create unique combination plot
    unique_combination_plot = create_unique_combination_plot(df_unique)
    return render_template('upload.html', demand_simulation_results=demand_simulation_results, demand_simulation_file_path=excel_path, plot_path=plot_path,plot_html_2=plot_html_2, unique_combination_plot=unique_combination_plot)

@app.route('/countrywise')
def countrywise_breakdown():
    # Run the optimization model
    status_out, objective_out, y, x, fix, var = optimization_model(fixed_costs, var_cost, demand, demand.columns[0], cap)
    
    # Initialize a dictionary to hold the flow data by country
    country_flow = {}

    # Iterate through production decisions
    for (i, j) in x.keys():
        if x[(i, j)].varValue > 0:  # Only include if there's production
            production_amount = x[(i, j)].varValue
            production_cost = var_cost.loc[i, j] * production_amount
            
            # Add the flow data to the respective country
            if j not in country_flow:
                country_flow[j] = {'amount': {}, 'cost': {}}
            if i not in country_flow[j]['amount']:
                country_flow[j]['amount'][i] = 0
                country_flow[j]['cost'][i] = 0
            
            country_flow[j]['amount'][i] += production_amount
            country_flow[j]['cost'][i] += production_cost
    
    # Prepare the breakdown for rendering
    breakdown_info = []
    
    for country, flows in country_flow.items():
        for source, amount in flows['amount'].items():
            cost = flows['cost'][source]
            breakdown_info.append((source, country, amount, cost))
    
    # Print the breakdown_info for debugging
    print(breakdown_info)  # Debugging line

    # Check if breakdown_info has correct shape
    if not breakdown_info:  # If no data is available
        breakdown_info = [("No data", "No data", "No data", "No data")]
    
    # Create a DataFrame for plotting
    df_breakdown = pd.DataFrame(breakdown_info, columns=['Source', 'Country', 'Amount', 'Cost'])
    
    # Create stacked graphs for amounts and costs
    amount_fig = create_stacked_graph(df_breakdown, 'Amount', list(country_flow.keys()), 'Amount Received by Country')
    cost_fig = create_stacked_graph(df_breakdown, 'Cost', list(country_flow.keys()), 'Cost of Products Received by Country')

    # Render the template with the breakdown information and plots
    return render_template('upload.html', amount_fig=amount_fig, cost_fig=cost_fig)

def create_stacked_graph(df, value_col, countries, title):
    # Pivot the DataFrame for stacked bar chart
    pivot_df = df.pivot_table(index='Country', columns='Source', values=value_col, aggfunc='sum', fill_value=0)
    
    # Create the stacked bar chart
    fig = go.Figure()
    for col in pivot_df.columns:
        fig.add_trace(go.Bar(
            x=pivot_df.index,
            y=pivot_df[col],
            name=col,
            hoverinfo='y+name',
            text=pivot_df[col],
            textposition='auto'
        ))

    fig.update_layout(
        paper_bgcolor='lightyellow',
        plot_bgcolor='lightyellow',
        title=title,
        barmode='stack',
        xaxis_title='Country',
        yaxis_title=value_col,
        height=600,
        width=900,
    )
    fig.update_yaxes(type="log")

    # Save the figure as an HTML string
    return fig.to_html(full_html=False)

@app.route('/submit_new_files', methods=['GET','POST'])
def upload_new_files():
    if request.method == 'POST':
        global uploaded_new_data  # Use the global variable to store uploaded data

        if 'files' not in request.files:
            return 'No file part'
        
        files = request.files.getlist('files')

        if not files or all(file.filename == '' for file in files):
            return 'No selected files'
        
        uploaded_files_info = []

        for file in files:
            if file and allowed_file(file.filename):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                data = pd.read_excel(file_path)

                # Set the first column as the index
                data.set_index(data.columns[0], inplace=True)

                # Store the DataFrame in the global dict
                uploaded_new_data[file.filename] = data  
                uploaded_files_info.append(f'File: {file.filename}<br>Data:<br>{data.head().to_html()}')
            else:
                uploaded_files_info.append(f'Invalid file type for {file.filename}')
        
        # Pass the preview HTML to the template
        return render_template('upload.html', uploaded_data=None,show_upload_new_form=True)
    else:
        return render_template('upload.html', show_upload_new_form=True)
    
@app.route('/preview_new')
def preview_new_files(): 
    global uploaded_new_data
    preview_info = ''  # String to store the combined HTML preview
    
    # Generate preview for each uploaded file
    for filename, data in uploaded_new_data.items():
        # Reset the index to include it in the HTML output
        data_preview = data.reset_index().head().to_html(classes='table table-striped table-bordered', index=False)

        # Combine everything into preview_info
        preview_info += f'''
            <div class="mb-4">
                <h2 class="text-danger">{filename}</h2>
                <div class="table-responsive">
                    {data_preview}
                </div>
            </div>
        '''
    
    # Now add the CO2 tax table to the preview
    global co2_tax_table,tax_table
    tax_table,co2_tax_table = CO2_tax()  # This returns the CO2 tax table in HTML format

    # Append the CO2 tax table to preview_info
    preview_info += f'''
        <div class="mb-4">
            <h2 class="text-center text-success">CO2 Tax Table</h2>
            <div class="table-responsive">
                {co2_tax_table}
            </div>
        </div>
    '''

    # Render the HTML using render_template, passing the preview_info
    return render_template('upload.html', uploaded_new_data=None, preview_new_info=preview_info, show_upload_form=False)

def calculate_tax(co2_value, baseline_value):
    if co2_value > baseline_value:
        return ((co2_value - baseline_value) // 0.2) *5  # 5% tax for every 0.1 tonne increase
    else:
        return 0

def CO2_tax():
    global uploaded_new_data
    global fixed_costs_EU, var_cost_EU, freight_costs_EU, cap_EU,CO2
    filenames = list(uploaded_new_data.keys())

    # Assign files based on matching filenames or indices
    for index, filename in enumerate(filenames):
        if index == 0:
            fixed_costs_EU = uploaded_new_data[filename]
        elif index == 1:
            var_cost_EU = uploaded_new_data[filename]
        elif index == 2:
            freight_costs_EU = uploaded_new_data[filename]
        elif index == 3:
            cap_EU = uploaded_new_data[filename]
        elif index == 4:
            CO2 = uploaded_new_data[filename]
        
    freight_costs_EU=freight_costs_EU/1000
    # Check if CO2 has enough columns
    if CO2.shape[1] < 3:
        return "Error: CO2 DataFrame does not have enough columns."

    # Create a new DataFrame to store tax rates for each country
    tax_rates = pd.DataFrame(columns=['Country', 'Tax Rate'])

    # Use the existing columns to calculate tax rates for each country
    for country in CO2.index:
        borderline_value = CO2.loc[country, CO2.columns[0]] + CO2.loc[country, CO2.columns[2]]
        baseline_co2 = CO2.iloc[1,0] + CO2.iloc[1,2]
        tax_rate = calculate_tax(borderline_value, baseline_co2)
        
        # Append the country and its tax rate to the new DataFrame using pd.concat
        new_row = pd.DataFrame({'Country': [country], 'Tax Rate': [tax_rate]})
        tax_rates = pd.concat([tax_rates, new_row], ignore_index=True)

    return tax_rates,tax_rates.to_html(classes='table table-striped table-bordered')

def execute_notebook_and_get_results(notebook_path, output_json_path):
    os.system(f'python run_notebook.py')  # Call the script to run the notebook
    with open(output_json_path, 'r') as file:
        results = json.load(file)
    return results
@app.route('/summarize_tax')
def summarize_tax_data():
    with open("C:\\Users\\Hp-D\\OneDrive\\Desktop\\supplyapp\\flask-server\\api\\templates\\optimization_results.json") as f:
        results = json.load(f)
    with open("C:\\Users\\Hp-D\\OneDrive\Desktop\\supplyapp\\flask-server\\api\\templates\\optimization_results_1.json") as f:
        results_1 = json.load(f)
    combined_fig = create_combined_plot()
    
    # Convert the figure to HTML
    plot_html = combined_fig.to_html(full_html=False)

    # Pass the results and plot to the HTML template
    return render_template('upload.html', results=results, results_1=results_1, plot_html=plot_html)

def create_combined_plot():
    # Factory locations with coordinates
    location_coords = {
        'USA': [-95.7129, 37.0902],
        'GERMANY': [10.4515, 51.1657],
        'JAPAN': [138.2529, 36.2048],
        'BRAZIL': [-51.9253, -14.2350],
        'INDIA': [78.9629, 20.5937]
    }

    # Example JSON data without optimization
    json_data_no_opt = '''
    {
        "status": "Optimal",
        "objective_value": 618310000.0,
        "fixed_costs": 18410000.0,
        "variable_costs": 4900000.0,
        "freight_cost": 595000000.0,
        "tax": 172060000.0,
        "plant_decisions": {
            "('USA', 'LOW')": 0.0,
            "('GERMANY', 'LOW')": 1.0,
            "('JAPAN', 'LOW')": 0.0,
            "('BRAZIL', 'LOW')": 0.0,
            "('INDIA', 'LOW')": 0.0,
            "('USA', 'HIGH')": 0.0,
            "('GERMANY', 'HIGH')": 1.0,
            "('JAPAN', 'HIGH')": 0.0,
            "('BRAZIL', 'HIGH')": 0.0,
            "('INDIA', 'HIGH')": 1.0
        },
        "production_quantities": {
            "('USA', 'Italy')": 0.0,
            "('GERMANY', 'Italy')": 300000.0,
            "('JAPAN', 'Italy')": 0.0,
            "('BRAZIL', 'Italy')": 0.0,
            "('INDIA', 'Italy')": 200000.0
        }
    }
    '''

    # Example JSON data with optimization
    json_data_opt = '''
    {
        "status": "Optimal",
        "objective_value": 748690000.0,
        "fixed_costs": 21750000.0,
        "variable_costs": 5700000.0,
        "freight_cost": 665000000.0,
        "tax": 56240000.0,
        "plant_decisions": {
            "('USA', 'LOW')": 0.0,
            "('GERMANY', 'LOW')": 1.0,
            "('JAPAN', 'LOW')": 0.0,
            "('BRAZIL', 'LOW')": 0.0,
            "('INDIA', 'LOW')": 0.0,
            "('USA', 'HIGH')": 1.0,
            "('GERMANY', 'HIGH')": 1.0,
            "('JAPAN', 'HIGH')": 0.0,
            "('BRAZIL', 'HIGH')": 0.0,
            "('INDIA', 'HIGH')": 0.0
        },
        "production_quantities": {
            "('USA', 'Italy')": 200000.0,
            "('GERMANY', 'Italy')": 300000.0,
            "('JAPAN', 'Italy')": 0.0,
            "('BRAZIL', 'Italy')": 0.0,
            "('INDIA', 'Italy')": 0.0
        }
    }
    '''
    # Load data
    data_no_opt = json.loads(json_data_no_opt)
    data_opt = json.loads(json_data_opt)

    # Create an empty figure
    fig = go.Figure() 

    # Function to add factories and connections to the figure
    def add_factory_data(data, title_suffix, legend_group):
        plant_decisions = data['plant_decisions']
        production_quantities = data['production_quantities']

        # Add markers for factory locations
        for loc, coord in location_coords.items():
            low_capacity = plant_decisions.get(f"('{loc}', 'LOW')", 0)
            high_capacity = plant_decisions.get(f"('{loc}', 'HIGH')", 0)

            # Determine marker color
            if low_capacity == 0 and high_capacity == 0:
                marker_color = 'red'  # Closed
            elif low_capacity == 1 and high_capacity == 0:
                marker_color = 'blue'  # Open with low capacity
            elif low_capacity == 0 and high_capacity == 1:
                marker_color = 'green'  # Open with high capacity
            elif low_capacity == 1 and high_capacity == 1:
                marker_color = 'purple'  # Both capacities open

            # Add factory marker
            fig.add_trace(go.Scattergeo(
                lon=[coord[0]], lat=[coord[1]],
                text=loc,
                hovertext=f"{loc}<br>Production to Italy: {production_quantities.get((loc, 'Italy'), 0)} units",
                mode='markers+text',
                marker=dict(size=10, color=marker_color),
                name=f'{loc} Factory',
                legendgroup=legend_group  # Group by optimization status
            )) 

            # Add production connections to Italy only
            production_qty = production_quantities.get(f"('{loc}', 'Italy')", 0)
            if production_qty > 0:  # Only draw lines for factories producing to Italy
                lon_start, lat_start = location_coords[loc]
                # Simulate a line to Italy's coordinates
                lon_italy, lat_italy = 12.5674, 41.8719  # Approximate coordinates of Italy

                # Create curvature for each line
                lon_curve = np.linspace(lon_start, lon_italy, 500)
                lat_curve = np.linspace(lat_start, lat_italy, 500) + np.sin(np.linspace(0, np.pi, 500)) * np.random.uniform(1, 20)

                # Use a unique color for each connection
                line_color = 'rgba(0, 0, 255, 0.5)'  # Blue color for lines to Italy

                # Add line trace with hover information
                fig.add_trace(go.Scattergeo(
                    lon=lon_curve,
                    lat=lat_curve,
                    mode='lines',
                    line=dict(width=2, color=line_color),
                    hoverinfo='text',
                    text=f'{loc} to Italy - Production: {production_qty} units',
                    name=f'{loc} to Italy',
                    legendgroup=legend_group  # Group by optimization status
                )) 

    # Add both data sets to the figure
    add_factory_data(data_no_opt, "(Without Optimization)", "Non-Optimized")
    add_factory_data(data_opt, "(With Optimization)", "Optimized")

    # Update layout for better segregation in the legend
    fig.update_layout(
        title="Factory Locations and Production Connections",
        geo=dict(
            projection_type="orthographic", 
            showcoastlines=True, 
            coastlinecolor="black", 
            landcolor="lightgreen", 
            showocean=True, 
            oceancolor="lightblue", 
            showcountries=True, 
            countrycolor='black',
            resolution=110,
        ),
        showlegend=True,
        legend=dict(
            title="Optimization Status",
            itemsizing='constant',  # Keep items the same size
            traceorder="normal",
            bgcolor='lightgray',  # Background color for legend
            bordercolor='black',
            borderwidth=1,
            font=dict(size=10)
        )
    )

    # Add custom text entries for visual separation
    fig.add_trace(go.Scattergeo(
        lon=[None], lat=[None],
        mode='text',
        text=['Non-Optimized Results'],
        textposition='bottom center',
        showlegend=True,
        name='Non-Optimized Header',
        legendgroup='Non-Optimized'
    ))

    fig.add_trace(go.Scattergeo(
        lon=[None], lat=[None],
        mode='text',
        text=['Optimized Results'],
        textposition='bottom center',
        showlegend=True,
        name='Optimized Header',
        legendgroup='Optimized'
    ))
    fig.update_layout(
        annotations=[
            dict(
                x=0, y=0.85,
                xref='paper', yref='paper',
                text='Factory Status:',  # Main title for the custom legend
                showarrow=False,
                font=dict(size=14, color="black"),
            ),
            dict(
                x=0, y=0.75,
                xref='paper', yref='paper',
                text='<span style="color:red;">●</span> Closed Factory',
                showarrow=False,
                font=dict(size=12),
            ),
            dict(
                x=0, y=0.65,
                xref='paper', yref='paper',
                text='<span style="color:blue;">●</span> Low Capacity Factory',
                showarrow=False,
                font=dict(size=12),
            ),
            dict(
                x=0, y=0.57,
                xref='paper', yref='paper',
                text='<span style="color:green;">●</span> High Capacity Factory',
                showarrow=False,
                font=dict(size=12),
            ),
            dict(
                x=0, y=0.50,
                xref='paper', yref='paper',
                text='<span style="color:purple;">●</span> Both Capacities Factory',
                showarrow=False,
                font=dict(size=12),
            ),
        ]
    )

    return fig

def optimization_model_with_tax(fixed_costs_EU, var_cost_EU, freight_costs_EU, cap_EU, co2_tax_rates):
    '''Optimization model considering carbon tax for shipping to a single EU destination (Italy)'''
    
    # Define Locations
    loc = ['USA', 'GERMANY', 'JAPAN', 'BRAZIL', 'INDIA']
    size = ['LOW', 'HIGH']
    destination_loc = ['Italy']  # Single destination for all shipments

    # Define Plant and Production Variables
    plant_name = [(i, s) for s in size for i in loc]
    prod_name = [(i, j) for i in loc for j in destination_loc]  # Each country ships to Italy
    co2_tax_rates.set_index(co2_tax_rates.columns[0], inplace=True)
    # Initialize Model
    model = LpProblem("Capacitated Plant Location Model with Carbon Tax", LpMinimize)

    # Create Decision Variables
    x = LpVariable.dicts("production_", prod_name, lowBound=0, upBound=None, cat='continuous')
    y = LpVariable.dicts("plant_", plant_name, cat='Binary')

    # Define Objective Function (with carbon tax on freight costs)
    model += (
        # Fixed costs
        lpSum([fixed_costs_EU.loc[i, s] * y[(i, s)] * 1000 for s in size for i in loc]) +
        # Variable costs
        lpSum([var_cost_EU.loc[i, var_cost_EU.columns[0]] * x[(i, 'Italy')] for i in loc]) +
        # Freight costs
        lpSum([freight_costs_EU.loc[i, freight_costs_EU.columns[0]] * x[(i, 'Italy')] for i in loc]) +
        # Carbon tax applied individually for each country's total cost
        lpSum([(var_cost_EU.loc[i, var_cost_EU.columns[0]] + freight_costs_EU.loc[i, freight_costs_EU.columns[0]]) 
                * co2_tax_rates.loc[i, co2_tax_rates.columns[0]] * x[(i, 'Italy')] / 100 for i in loc])
    )

    # Add Constraints
    
    model += lpSum([x[(i, 'Italy')] for i in loc]) == 500000  # Italy's total demand

    for i in loc:
        model += lpSum(x[(i, 'Italy')]) <= lpSum([cap_EU.loc[i, s] * y[(i, s)] * 1000 for s in size])  # Capacity limits per country
    
    # Solve Model
    model.solve()

    # Results
    status_out = LpStatus[model.status]
    objective_out = pulp.value(model.objective)
    fix = sum([fixed_costs_EU.loc[i, s] * y[(i, s)].varValue * 1000 for s in size for i in loc])
    var = sum([var_cost_EU.loc[i, var_cost_EU.columns[0]] * x[(i, 'Italy')].varValue for i in loc])
    freight = sum([freight_costs_EU.loc[i, freight_costs_EU.columns[0]] * x[(i, 'Italy')].varValue for i in loc])
    
    # Calculate tax after solving the model
    tax = sum([(var_cost_EU.loc[i, var_cost_EU.columns[0]] + freight_costs_EU.loc[i, freight_costs_EU.columns[0]])  
                * co2_tax_rates.loc[i, co2_tax_rates.columns[0]] * x[(i, 'Italy')].varValue / 100 for i in loc])

    return status_out, objective_out, y, x, fix, var, freight, tax

def optimization_model_without_tax(fixed_costs_EU, var_cost_EU, freight_costs_EU, cap_EU, co2_tax_rates_1):
    '''Optimization model without considering carbon tax initially. CO2 tax is calculated after optimization.'''

    loc = ['USA', 'GERMANY', 'JAPAN', 'BRAZIL', 'INDIA']
    size = ['LOW', 'HIGH']
    destination_loc = ['Italy']  # Single destination for all shipments
    co2_tax_rates_1.set_index(co2_tax_rates_1.columns[0],inplace=True)
    logging.debug(f"CO2 Tax Rates DataFrame:\n{co2_tax_rates_1}")

    # Define Plant and Production Variables
    plant_name = [(i, s) for s in size for i in loc]
    prod_name = [(i, j) for i in loc for j in destination_loc]  # Each country ships to Italy
    
    # Initialize Model
    model = LpProblem("Capacitated Plant Location Model without Carbon Tax", LpMinimize)

    # Create Decision Variables
    x = LpVariable.dicts("production_", prod_name, lowBound=0, upBound=None, cat='continuous')
    y = LpVariable.dicts("plant_", plant_name, cat='Binary')

    # Define Objective Function (without carbon tax)
    model += (
    # Fixed costs
    lpSum([fixed_costs_EU.loc[i, s] * y[(i, s)] * 1000 for s in size for i in loc])
    # Variable costs
    + lpSum([var_cost_EU.loc[i, var_cost_EU.columns[0]] * x[(i, 'Italy')] for i in loc])
    # Freight costs
    + lpSum([freight_costs_EU.loc[i, freight_costs_EU.columns[0]] * x[(i, 'Italy')] for i in loc])
    )

    # Add Constraints
    for j in destination_loc:
        model += lpSum([x[(i, j)] for i in loc]) == 500000  # Italy's total demand

    for i in loc:
        model += lpSum(x[(i, 'Italy')]) <= lpSum([cap_EU.loc[i, s] * y[(i, s)] * 1000 for s in size])  # Capacity limits per country
    
    # Solve Model
    model.solve()

    # Results without CO2 Tax
    status_out = LpStatus[model.status]
    objective_out = pulp.value(model.objective)
    fix = sum([fixed_costs_EU.loc[i, s] * y[(i, s)].varValue * 1000 for s in size for i in loc])
    var = sum([var_cost_EU.loc[i,var_cost_EU.columns[0]] * x[(i, 'Italy')].varValue for i in loc])
    freight = sum([freight_costs_EU.loc[i,freight_costs_EU.columns[0]] * x[(i, 'Italy')].varValue for i in loc])

    # Calculate the CO2 tax separately, after solving the model
    tax = sum([(var_cost_EU.loc[i, var_cost_EU.columns[0]] 
            + freight_costs_EU.loc[i, freight_costs_EU.columns[0]])  
            * co2_tax_rates_1.loc[i, co2_tax_rates_1.columns[0]] 
            * x[(i, 'Italy')].varValue / 100 for i in loc])

    return status_out, objective_out,y, x, fix, var, freight, tax

@app.route('/ESG Score', methods=['GET', 'POST'])
def upload_esg_table():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        global esg_score
        
        if file and file.filename.endswith('.xlsx'):
            # Read the Excel file using pandas
            excel_data = pd.read_excel(file)

            # Convert the uploaded Excel data to HTML with Bootstrap classes
            uploaded_data_html = excel_data.to_html(classes='table table-striped table-bordered', index=False)

            # Initialize an empty DataFrame to store ESG scores
            esg_score = pd.DataFrame()

            # Calculate ESG score for each country (columns 2 to 7)
            for j in range(2, 7):  # Looping over the columns for the countries
                score = 0
                for i in range(0, 6):  # Looping over the ESG criteria rows
                    score += excel_data.iloc[i, 1] * excel_data.iloc[i, j]  # Weight * Score for each country
                
                # Create a DataFrame for the country's ESG score and transpose it
                country_score = pd.DataFrame({excel_data.columns[j]: [score/10]})
                
                # Concatenate the new score into the esg_score DataFrame
                esg_score = pd.concat([esg_score, country_score], axis=1)

            # Reset the index to have a clean result
            esg_score.reset_index(drop=True, inplace=True)

            # Convert the ESG score DataFrame to HTML with Bootstrap classes
            esg_plot = esg_score.to_html(classes='table table-striped table-bordered', index=True)

            # Set show_upload_form_1 to False to hide the upload form after a file is uploaded
            return render_template('upload.html', uploaded_data_1=uploaded_data_html, show_upload_form_1=False, esg_plot=esg_plot)
    
    # For GET request, just render the upload form
    return render_template('upload.html', uploaded_data_1='', show_upload_form_1=True)

@app.route('/ESG Optimization')
def ESG_Optimization():
    esg_scores = pd.DataFrame({
        'Country': ['USA', 'GERMANY', 'JAPAN', 'BRAZIL', 'INDIA'],
        'ESG_Score': [8.15, 8.65, 7.65, 6.3, 5.8]
    })

    # Initialize an empty DataFrame to store ESG tax values
    esg_tax = pd.DataFrame(columns=['Country', 'Tax %'])

    # Iterate through the ESG scores to calculate the tax for each country
    for i, row in esg_scores.iterrows():
        esg_score = row['ESG_Score']  # Get the ESG score for the country
        tax_value = max((8 - esg_score) * 5, 0)  # Calculate the tax based on the ESG score logic

        # Create a new row as a DataFrame
        new_row = pd.DataFrame({'Country': [row['Country']], 'Tax %': [tax_value]})

        # Concatenate the new row with esg_tax
        esg_tax = pd.concat([esg_tax, new_row], ignore_index=True)

    # Render the results in the template
    return render_template('upload.html', esg_tax=esg_tax.to_html(classes='table table-striped table-bordered', index=False))

if __name__ == '__main__':
    app.run(debug=True)
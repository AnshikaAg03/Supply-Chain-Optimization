from flask import Flask, request, render_template, redirect, url_for
from flask_cors import CORS  # Import Flask-CORS
import os
import pandas as pd
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, value, LpStatus
import pulp
import plotly.graph_objects as go
from plotly.offline import plot
import numpy as np
import plotly.io as pio


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes by default

UPLOAD_FOLDER = 'uploads/'  # Ensure this folder exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Store uploaded files data for use across routes
uploaded_data = {}

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'xlsx', 'xls'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
        # Get column names, index names, and data types
        column_names = data.columns.tolist()
        index_names = data.index.names if data.index.names[0] is not None else ['Index']
        data_types = data.dtypes.to_frame(name='Data Type')
        
        # Generate HTML for column names, index names, and data types
        column_info = f"<strong>Column Names:</strong> {', '.join(column_names)}<br>"
        index_info = f"<strong>Index Names:</strong> {', '.join(index_names)}<br>"
        dtype_info = f"<strong>Data Types:</strong><br>{data_types.to_html()}<br>"

        # Add the data preview (first few rows of the DataFrame)
        data_preview = data.head().to_html() 

        # Combine everything into preview_info
        preview_info += f'<h2>{filename}</h2>{column_info}{index_info}{dtype_info}{data_preview}<br><br>'
    
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
            demand = uploaded_data[filename]
        elif index==1:
            cap=uploaded_data[filename]
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
    excel_path = f"C:\\Users\\spart\\Desktop\\Datascience\\Analytics Vidya\\uploads\\df_demand-{int(CV * 100)}PC.xlsx"
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
    df_bool.to_excel(f'C:\\Users\\spart\\Desktop\\Datascience\\Analytics Vidya\\uploads\\boolean-{int(CV * 100)}PC.xlsx')

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

    return render_template('upload.html', demand_simulation_results=demand_simulation_results, demand_simulation_file_path=excel_path, plot_path=plot_path,plot_html_2=plot_html_2)

@app.route('/countrywise')
def countrywise_breakdown():
    global uploaded_data
    countrywise_info = ''
    for filename, data in uploaded_data.items():
        if 'Country' in data.columns:
            breakdown = data['Country'].value_counts().to_frame()
            breakdown.columns = ['Count']
            countrywise_info += f'<h2>Countrywise Breakdown for {filename}</h2>{breakdown.to_html()}<br><br>'
        else:
            countrywise_info += f'<h2>No Country Data for {filename}</h2><br>'
    return render_template('upload.html', uploaded_data=countrywise_info, show_upload_form=False)

if __name__ == '__main__':
    app.run(debug=True)

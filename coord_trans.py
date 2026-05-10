import numpy as np  # Importing numpy for mathematical operations, like handling arrays, and working with numbers.
import pandas as pd  # Importing pandas for data manipulation and handling data frames.
import json  # Importing json for working with JSON data (loading and saving JSON format).
import re  # Importing re for regular expression operations (for pattern matching).
import matplotlib.pyplot as plt  # Importing matplotlib for plotting graphs and visualizations.
from sklearn.ensemble import RandomForestClassifier  # Importing RandomForestClassifier from sklearn for classification tasks.
from sklearn.model_selection import train_test_split  # Importing train_test_split to split the dataset into training and testing subsets.
from sklearn.metrics import accuracy_score  # Importing accuracy_score to evaluate the accuracy of the model.

# -------------------------
# Feature extraction
# -------------------------
def extract_features(description):  # Function to extract features from a given region description.
    features = {  # A dictionary to store the extracted features.
        'sphere': 'sphere' in description.lower() or 'hemisphere' in description.lower(),  # Check if "sphere" or "hemisphere" is in the description.
        'cylinder': 'cylinder' in description.lower(),  # Check if "cylinder" is in the description.
        'cone': 'cone' in description.lower(),  # Check if "cone" is in the description.
        'paraboloid': 'paraboloid' in description.lower(),  # Check if "paraboloid" is in the description.
        'ellipsoid': 'ellipsoid' in description.lower(),  # Check if "ellipsoid" is in the description.
        'x^2+y^2': bool(re.search(r'x\^2\s*\+\s*y\^2', description)),  # Check if the pattern 'x^2 + y^2' exists in the description using regex.
        'x^2+y^2+z^2': bool(re.search(r'x\^2\s*\+\s*y\^2\s*\+\s*z\^2', description)),  # Check for 'x^2 + y^2 + z^2' using regex.
        'bounded_z': 'bounded between z' in description.lower()  # Check if "bounded between z" exists in the description.
    }
    return features  # Return the dictionary of extracted features.

# -------------------------
# Model training
# -------------------------
def train_model(dataset_path='expanded_volume_dataset.json'):  # Function to train the model using a dataset located at the specified path.
    with open(dataset_path, 'r') as f:  # Open the dataset file in read mode.
        data = json.load(f)  # Load the JSON data into a Python dictionary.

    descriptions = [item['desc'] for item in data]  # Extract descriptions of the regions from the dataset.
    coordinates = [item['coord'] for item in data]  # Extract the coordinate systems for each region from the dataset.
    feature_dicts = [extract_features(desc) for desc in descriptions]  # Extract features from each description.
    df = pd.DataFrame(feature_dicts)  # Convert the extracted features into a pandas DataFrame for easier processing.

    X = df  # Set the DataFrame as input features (X).
    y = coordinates  # Set the coordinates as the target labels (y).
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data into training and testing sets.

    model = RandomForestClassifier(n_estimators=100, random_state=42)  # Create a RandomForestClassifier model with 100 trees.
    model.fit(X_train, y_train)  # Train the model using the training data.

    y_pred = model.predict(X_test)  # Use the model to predict the coordinates for the test data.
    accuracy = accuracy_score(y_test, y_pred)  # Calculate the accuracy of the model by comparing predicted and actual values.
    print(f"\n✅ Model trained! Accuracy: {accuracy:.2f}")  # Print the accuracy of the model.

    return model  # Return the trained model.

# -------------------------
# Coordinate suggestion
# -------------------------
def suggest_coordinate_system(description, model):  # Function to suggest a coordinate system based on the region description.
    features = extract_features(description)  # Extract features from the given description.
    feature_df = pd.DataFrame([features])  # Convert the features into a DataFrame.
    suggestion = model.predict(feature_df)[0]  # Use the trained model to predict the coordinate system for the given features.
    proba = model.predict_proba(feature_df)[0]  # Get the probabilities for each possible coordinate system.
    confidence = max(proba)  # Calculate the confidence level as the maximum probability.

    return {  # Return a dictionary with the suggested coordinate system and confidence.
        'suggested_system': suggestion,
        'confidence': confidence,
        'recommendation': f"For computing the volume of this region, {suggestion} coordinates are recommended (confidence: {confidence:.2f})"
    }

# -------------------------
# Integral setup verification
# -------------------------
def verify_integral_setup(description, coord_system, integral_text, model):  # Function to verify if the integral setup matches the suggested system.
    suggestion = suggest_coordinate_system(description, model)  # Get the suggested coordinate system based on the description.
    coord_match = suggestion['suggested_system'] == coord_system  # Check if the provided coordinate system matches the suggested one.

    valid_setup = True  # Flag to track if the integral setup is valid.
    desc_lower = description.lower()  # Convert description to lowercase for case-insensitive checks.

    if coord_system == 'cylindrical':  # If the coordinate system is cylindrical.
        if 'sphere' in desc_lower and r'\rho' in integral_text:  # If the description mentions a sphere but the integral uses cylindrical variables, it's invalid.
            valid_setup = False
        if 'cylinder' in desc_lower and not 'r' in integral_text:  # If the description mentions a cylinder but the integral does not include 'r', it's invalid.
            valid_setup = False

    if coord_system == 'spherical':  # If the coordinate system is spherical.
        if 'sphere' in desc_lower and not r'\rho^2 sin(\phi)' in integral_text:  # If the description mentions a sphere but the integral does not match spherical coordinates, it's invalid.
            valid_setup = False

    result = {  # Prepare the result dictionary.
        'valid_setup': valid_setup and coord_match,  # Check if both the setup is valid and the coordinate system matches.
        'recommended_system': suggestion['suggested_system'],
        'message': ''  # Initialize an empty message.
    }

    if not coord_match:  # If the coordinate systems do not match.
        result['message'] = f"❗ Consider using {suggestion['suggested_system']} coordinates instead."  # Suggest the recommended system.
    elif not valid_setup:  # If the setup is invalid.
        result['message'] = "⚠️ There may be issues with your integral setup. Check the integrand and limits."  # Warn about issues with the setup.
    else:  # If everything is correct.
        result['message'] = "✅ The integral setup appears correct."  # Confirm that the setup is valid.

    return result  # Return the result with validation and a message.

# -------------------------
# Volume estimation
# -------------------------
def estimate_volume(integral_text):  # Function to estimate volume based on the given integral text.
    try:
        if 'cone' in integral_text.lower():  # Check if the description mentions a cone.
            radius_match = re.search(r'radius\s+(\d+)', integral_text)  # Try to find the radius in the integral text using regex.
            height_match = re.search(r'height\s+(\d+)', integral_text)  # Try to find the height in the integral text using regex.
            if radius_match and height_match:  # If both radius and height are found.
                r = float(radius_match.group(1))  # Convert the radius to float.
                h = float(height_match.group(1))  # Convert the height to float.
                volume = (1/3) * np.pi * r**2 * h  # Calculate the volume of the cone using the formula.
                return {"volume": volume, "message": f"Estimated volume: {volume:.4f}"}  # Return the volume and a message.

        if 'sphere' in integral_text.lower():  # Check if the description mentions a sphere.
            radius_match = re.search(r'radius\s+(\d+)|r\s*=\s*(\d+)', integral_text)  # Find the radius of the sphere.
            if radius_match:  # If the radius is found.
                r = float(radius_match.group(1) or radius_match.group(2))  # Get the radius value and convert it to float.
                volume = (4/3) * np.pi * r**3  # Calculate the volume of the sphere using the formula.
                return {"volume": volume, "message": f"Estimated volume: {volume:.4f}"}  # Return the volume and a message.

        if 'cylinder' in integral_text.lower():  # Check if the description mentions a cylinder.
            radius_match = re.search(r'radius\s+(\d+)|r\s*=\s*(\d+)', integral_text)  # Find the radius of the cylinder.
            z_bounds = re.findall(r'z from (-?\d+) to (-?\d+)', integral_text)  # Find the bounds of z (height).
            if radius_match and z_bounds:  # If both radius and z bounds are found.
                r = float(radius_match.group(1) or radius_match.group(2))  # Get the radius value.
                z_min = float(z_bounds[0][0])  # Get the lower bound of z.
                z_max = float(z_bounds[0][1])  # Get the upper bound of z.
                height = z_max - z_min  # Calculate the height of the cylinder.
                volume = np.pi * r**2 * height  # Calculate the volume of the cylinder using the formula.
                return {"volume": volume, "message": f"Estimated volume: {volume:.4f}"}  # Return the volume and a message.

        return {"volume": None, "message": "❌ Question is beyond my scope. Consider using numerical integration."}  # If unable to estimate volume.

    except Exception as e:  # If there's an error during volume estimation.
        return {"volume": None, "message": f"❌ Unable to estimate volume: {str(e)}"}  # Return an error message.

# -------------------------
# Visualize region (enhanced)
# -------------------------
def visualize_region(description):  # Function to visualize the region described in the input.
    desc_lower = description.lower()  # Convert description to lowercase for case-insensitive checks.
    fig = plt.figure()  # Create a new figure for the plot.
    ax = fig.add_subplot(111, projection='3d')  # Add a 3D subplot to the figure.

    if 'sphere' in desc_lower or re.search(r'x\^2\s*\+\s*y\^2\s*\+\s*z\^2\s*=\s*\d+', description):  # Check for sphere.
        r = 3  # Set the radius of the sphere.
        u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]  # Create a grid for spherical coordinates.
        x = r * np.cos(u) * np.sin(v)  # Calculate the x-coordinates of the sphere.
        y = r * np.sin(u) * np.sin(v)  # Calculate the y-coordinates of the sphere.
        z = r * np.cos(v)  # Calculate the z-coordinates of the sphere.
        ax.plot_surface(x, y, z, color='skyblue', alpha=0.6)  # Plot the sphere surface.
        ax.set_title('Sphere')  # Set the title of the plot.

    elif 'cylinder' in desc_lower or re.search(r'x\^2\s*\+\s*y\^2\s*=\s*\d+', description):  # Check for cylinder.
        r = 3  # Set the radius of the cylinder.
        h = 6  # Set the height of the cylinder.
        z = np.linspace(0, h, 50)  # Create the z-values for the height of the cylinder.
        theta = np.linspace(0, 2*np.pi, 50)  # Create the angular values for the cylinder.
        theta, z = np.meshgrid(theta, z)  # Create a meshgrid of theta and z values.
        x = r * np.cos(theta)  # Calculate the x-coordinates of the cylinder.
        y = r * np.sin(theta)  # Calculate the y-coordinates of the cylinder.
        ax.plot_surface(x, y, z, color='lightgreen', alpha=0.6)  # Plot the cylinder surface.
        ax.set_title('Cylinder')  # Set the title of the plot.

    elif 'cone' in desc_lower:  # Check for cone.
        h = 5  # Set the height of the cone.
        r = 3  # Set the radius of the cone.
        z = np.linspace(0, h, 50)  # Create the z-values for the height of the cone.
        theta = np.linspace(0, 2*np.pi, 50)  # Create the angular values for the cone.
        theta, z = np.meshgrid(theta, z)  # Create a meshgrid of theta and z values.
        x = (r - r*z/h) * np.cos(theta)  # Calculate the x-coordinates of the cone.
        y = (r - r*z/h) * np.sin(theta)  # Calculate the y-coordinates of the cone.
        ax.plot_surface(x, y, z, color='salmon', alpha=0.6)  # Plot the cone surface.
        ax.set_title('Cone')  # Set the title of the plot.

    else:  # If the shape is not recognized.
        print("❌ Cannot visualize this shape.")  # Print an error message.
        return  # Return without plotting anything.

    ax.set_xlabel('X')  # Label the x-axis.
    ax.set_ylabel('Y')  # Label the y-axis.
    ax.set_zlabel('Z')  # Label the z-axis.
    plt.show()  # Display the plot.

# -------------------------
# Main Menu
# -------------------------
def main():  # Main function to run the tool.
    print("📦 Volume AI Tool")  # Print a greeting message.
    model = train_model()  # Train the model.

    while True:  # Start an infinite loop to display the menu repeatedly.
        print("\nWhat would you like to do?")  # Show the options to the user.
        print("1. Suggest coordinate system")
        print("2. Verify integral setup")
        print("3. Estimate volume")
        print("4. Visualize region")
        print("5. Exit")

        choice = input("Enter choice (1-5): ").strip()  # Get the user's menu choice.

        if choice == '1':  # If the user chose option 1.
            desc = input("Enter region description: ").strip()  # Get the region description from the user.
            result = suggest_coordinate_system(desc, model)  # Suggest a coordinate system.
            print(f"\n🔵 Suggested: {result['suggested_system']} (Confidence: {result['confidence']:.2f})")  # Print the suggestion.
            print(result['recommendation'])  # Print the recommendation message.

        elif choice == '2':  # If the user chose option 2.
            desc = input("Enter region description: ").strip()  # Get the region description.
            coord = input("Enter coordinate system used (e.g., cylindrical): ").strip()  # Get the coordinate system.
            integral = input("Enter your integral text: ").strip()  # Get the integral text.
            result = verify_integral_setup(desc, coord, integral, model)  # Verify the integral setup.
            print(f"\n{result['message']}")  # Print the verification result message.
            print(f"Recommended system: {result['recommended_system']}")  # Print the recommended system.

        elif choice == '3':  # If the user chose option 3.
            integral_text = input("Enter description with shape and dimensions (e.g., 'cone radius 3 height 5'): ").strip()  # Get the description.
            result = estimate_volume(integral_text)  # Estimate the volume.
            print(f"\n{result['message']}")  # Print the volume estimation result.

        elif choice == '4':  # If the user chose option 4.
            desc = input("Enter region description to visualize: ").strip()  # Get the region description for visualization.
            visualize_region(desc)  # Visualize the region.

        elif choice == '5':  # If the user chose option 5.
            print("👋 Goodbye!")  # Print a goodbye message.
            break  # Exit the loop.

        else:  # If the user entered an invalid choice.
            print("❌ Invalid choice. Please enter a number between 1 and 5.")  # Prompt the user to enter a valid choice.

# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":  # Ensures that the program runs only when executed directly (not when imported).
    main()

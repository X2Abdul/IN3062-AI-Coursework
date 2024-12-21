import random, os

from pathlib import Path


# --------------------------------------------------------------------------------------------


def generate_dirty_data(num_samples):

    # Define the realistic ranges for each of the variables

    r_temp_range = (18.0, 24.0)  # Temperature in Celsius
    r_humidity_range = (18.0, 45.0)  # Humidity in relative percentage
    r_light_range = (0.0, 530.0)  # Light in Lux
    r_co2_range = (350.0, 2000.0)  # CO2 in ppm
    r_humidity_ratio_range = (0.0025, 0.006)  # Humidity ratio

    # Define the extreme ranges for each of the variables

    ur_temp_range = (5.0, 35.0)  # Temperature in Celsius
    ur_humidity_range = (0.0, 70.0)  # Humidity in relative percentage
    ur_light_range = (0.0, 650.0)  # Light in Lux
    ur_co2_range = (50.0, 3000.0)  # CO2 in ppm
    ur_humidity_ratio_range = (0.002, 0.007)  # Humidity ratio

    # Array to hold entries

    data_rows = []

    # Generate the data

    for i in range(num_samples):

        # Random number
        
        id = f'"{random.randint(1, num_samples)}"'

        date = f'"{random.randint(1, num_samples)}"'

        # 23% chance of being in the realistic value range else is in the unrealistic value range, which includes the realistic value range

        temperature = round(random.uniform(*r_temp_range) if random.random() < 0.23 else random.uniform(*ur_temp_range), 5)

        humidity = round(random.uniform(*r_humidity_range) if random.random() < 0.23 else random.uniform(*ur_humidity_range), 5)

        light = round(random.uniform(*r_light_range) if random.random() < 0.23 else random.uniform(*ur_light_range), 2)

        co2 = round(random.uniform(*r_co2_range) if random.random() < 0.23 else random.uniform(*ur_co2_range), 2)

        humidity_ratio = round(random.uniform(*r_humidity_ratio_range) if random.random() < 0.23 else random.uniform(*ur_humidity_ratio_range), 10)

        # Equal chance of being a 0 or 1

        occupancy = random.choice([0, 1])

        data_row = f'{id},{date},{temperature},{humidity},{light},{co2},{humidity_ratio},{occupancy}'
        data_rows.append(data_row)

    return data_rows


def insert_data(input_file, input_data, has_header, shuffle):

    # Open file and write lines

    with open(input_file, 'a') as file:
        for line in input_data:
            file.write(f"{line}\n")
    
        file.close()

    if (shuffle):

        with open(input_file, 'r+') as file:
            lines = file.readlines()

        if (has_header):

            # Split header off

            vars_header = lines[0]
            data_lines = lines[1:]

            # Shuffle only the data lines and put together again

            random.shuffle(data_lines)
            lines = [vars_header] + data_lines

        else:

            # Only shuffle

            random.shuffle(lines)
            file.writelines(lines)
            
        file.close()

    print("Data inserted successfully")


# --------------------------------------------------------------------------------------------


# Test if in the correct working directory, else change current working directory

cwd = Path().resolve()

while True:

    cwd = Path().resolve()

    if not (cwd / "src").is_dir():
        os.chdir(cwd.parent)
    else:
        break

os.chdir(cwd / "datasets/usable")
cwd = Path().resolve()

# Generate the dirty data

samples = 10000
dirty_data = generate_dirty_data(samples)

# Save dirty data to file and shuffle

file = "datatraining.txt"

insert_data(file, dirty_data, True, True)
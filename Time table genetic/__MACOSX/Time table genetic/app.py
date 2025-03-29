import streamlit as st
import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
import random
from functools import partial

# Define data structures
class Teacher:
    def __init__(self, name):
        self.name = name
        self.subjects = []

    def assign_subject(self, subject):
        self.subjects.append(subject)

class Subject:
    def __init__(self, name):
        self.name = name

class Venue:
    def __init__(self, name):
        self.name = name

# Genetic Algorithm setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def generate_individual(teachers, subjects, venues, num_sections, hours_per_day):
    # Randomly assign teachers and venues for each section and hour
    individual = []
    for hour in range(hours_per_day):
        for section in range(num_sections):
            teacher = random.choice(teachers)
            subject = random.choice(teacher.subjects)
            venue = random.choice(venues)
            individual.append((teacher.name, subject.name, venue.name))
    return individual

def evaluate(individual):
    # Fitness function to evaluate the quality of the timetable
    penalty = 0
    
    # Check for teacher or venue clashes in the same hour
    schedule_by_hour = {}
    for idx, (teacher_name, subject_name, venue_name) in enumerate(individual):
        hour = idx // num_sections  # Determine which hour this entry belongs to
        
        if hour not in schedule_by_hour:
            schedule_by_hour[hour] = {'teachers': set(), 'venues': set()}
        
        if teacher_name in schedule_by_hour[hour]['teachers']:
            penalty += 1  # Teacher clash
        
        if venue_name in schedule_by_hour[hour]['venues']:
            penalty += 1  # Venue clash
        
        schedule_by_hour[hour]['teachers'].add(teacher_name)
        schedule_by_hour[hour]['venues'].add(venue_name)
    
    return penalty,

def mutate(individual):
    # Randomly swap teachers or venues in the timetable
    idx = random.randint(0, len(individual) - 1)
    if random.random() < 0.5:
        # Swap teacher
        individual[idx] = (random.choice(teachers).name, individual[idx][1], individual[idx][2])
    else:
        # Swap venue
        individual[idx] = (individual[idx][0], individual[idx][1], random.choice(venues).name)
    
    return individual,

def crossover(ind1, ind2):
    # Single point crossover between two individuals
    size = min(len(ind1), len(ind2))
    cxpoint = random.randint(1, size - 1)
    
    ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]
    
    return ind1, ind2

def generate_timetable_ga(teachers, subjects, venues, num_sections, hours_per_day):
    toolbox = base.Toolbox()
    
    # Use functools.partial to pass additional arguments to generate_individual
    toolbox.register("individual", tools.initIterate,
                     creator.Individual,
                     partial(generate_individual,
                             teachers=teachers,
                             subjects=subjects,
                             venues=venues,
                             num_sections=num_sections,
                             hours_per_day=hours_per_day))
    
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", evaluate)
    
    toolbox.register("mate", crossover)
    
    toolbox.register("mutate", mutate)
    
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=100)  # Population size of 100

    # Run the genetic algorithm
    algorithms.eaSimple(population, toolbox,
                        cxpb=0.7, mutpb=0.2,
                        ngen=50,  # Number of generations
                        verbose=False)

    best_individual = tools.selBest(population, k=1)[0]
    
    timetable_data = []
    
    for idx in range(len(best_individual)):
        hour = idx // num_sections + 1
        section_number = idx % num_sections + 1
        teacher_name, subject_name, venue_name = best_individual[idx]
        
        timetable_data.append({
            'Hour': hour,
            'Section': f'Section {section_number}',
            'Teacher Name': teacher_name,
            'Subject Name': subject_name,
            'Venue': venue_name
        })
    
    return pd.DataFrame(timetable_data)

# Streamlit UI setup (unchanged from your original code)
# Streamlit UI setup (unchanged from your original code)
st.title('Timetable Scheduling using Genetic Algorithm')

num_teachers = st.number_input('Number of Teachers', min_value=1)
num_sections = st.number_input('Number of Sections', min_value=1)
num_subjects = st.number_input('Number of Subjects', min_value=1)
num_classrooms = st.number_input('Number of Venues', min_value=1)
hours_per_day = st.number_input('Hours per Day', min_value=1)

# Validation for hours_per_day <= num_teachers
if hours_per_day > num_teachers:
    st.error("The number of hours per day cannot exceed the number of teachers.")

# Validation for number of venues >= number of sections
if num_classrooms < num_sections:
    st.error("The number of venues should be greater than or equal to the number of sections.")

else:
    teachers = []
    subjects = []
    venues = []

    # Input details for each subject (unchanged from your original code)
    st.header("Subjects Information")
    for i in range(num_subjects):
        with st.expander(f"Subject {i+1}"):
            name = st.text_input(f"Name of Subject {i+1}", key=f'subject_name_{i}')
            if name:
                subjects.append(Subject(name))

    # Input details for each teacher (unchanged from your original code)
    st.header("Teachers Information")
    for i in range(num_teachers):
        with st.expander(f"Teacher {i+1}"):
            name = st.text_input(f"Name of Teacher {i+1}", key=f'teacher_name_{i}')
            
            teacher = Teacher(name)
            
            subject_ids = st.multiselect(f"Subjects Teacher {i+1} can teach", [sub.name for sub in subjects], key=f'teacher_subjects_{i}')
            
            for subject_name in subject_ids:
                subject = next((sub for sub in subjects if sub.name == subject_name), None)
                if subject:
                    teacher.assign_subject(subject)
            
            teachers.append(teacher)

    # Input details for each venue (unchanged from your original code)
    st.header("Venues Information")
    for i in range(num_classrooms):
        with st.expander(f"Venue {i+1}"):
            name = st.text_input(f"Name of Venue {i+1}", key=f'venue_name_{i}')
            if name:
                venues.append(Venue(name))

    if st.button('Generate Timetable'):
        if teachers and subjects and venues:
            generated_timetable_ga_df = generate_timetable_ga(teachers, subjects, venues,
                                                              num_sections=num_sections,
                                                              hours_per_day=hours_per_day)
            
            st.write(generated_timetable_ga_df)
            
        else:
            st.error("Please ensure you have added teachers, subjects and venues before generating the timetable.")

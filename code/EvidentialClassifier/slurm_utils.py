import subprocess
import time
import os

def make_dirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def submit_slurm_job(command, message=""):
    output = subprocess.check_output(command, shell=True, text=True)
    job_id = output.strip().split()[-1]
    if len(message) != 0:
        print("--{}: {}--".format(message, job_id))
    return job_id

def slurm_generate_video(job_ids, config_file, output_path):
    command = "sbatch --dependency=afterok:{} ./vasp.generate_video.sh {} {}".format(
        ",".join(job_ids), config_file, output_path
    )
    job_id = submit_slurm_job(command, "Generate video")
    return job_id

def slurm_update_database(
    job_ids, config_file, prompts_file,
    data_file, output_path, step
):
    command = "sbatch --dependency=afterok:{} ./vasp.update_database.sh {} {} {} {} {}".format(
        ",".join(job_ids), config_file, prompts_file, 
        data_file, output_path, step
    )
    job_id = submit_slurm_job(command, "Update database")
    return job_id

def slurm_run_GAFS(
    prompts, config_file, data_file, 
    output_path
):
    job_ids = []
    make_dirs("{}/run".format(output_path))
    for prompt in prompts:
        command = "sbatch ./vasp.GAFS.sh {} {} '{}'".format(config_file, output_path, prompt)
        job_ids.append(submit_slurm_job(command, "Run GAFS with prompt {}".format(prompt)))
    return job_ids

def slurm_run_visualization_method(
    job_ids, config_file, prompts_file, 
    output_path, initial_data="random"
):
    if job_ids is None:
        command = "sbatch vasp.manifold_learning.sh {} {} {} {}".format(
            config_file, prompts_file, 
            output_path, initial_data
        )
    else:
        command = "sbatch --dependency=afterok:{} vasp.manifold_learning.sh {} {} {} {}".format(
            ",".join(job_ids), config_file, prompts_file, 
            output_path, initial_data
        )
    job_id = submit_slurm_job(command, "Manifold learning")
    return job_id

def slurm_model_substitution_method(
    config_file, prompts_file, output_path, 
    update_mask=False, mode="multiple"
):
    script_paths = {
        "single": {
            "distance_matrix_calculator": "vasp.distance_matrix_calculator_single.sh",
            "model_substitution_method": "vasp.model_substitution_method_single.sh",
        },
        "multiple": {
            "distance_matrix_calculator": "vasp.distance_matrix_calculator_multiple.sh",
            "model_substitution_method": "vasp.model_substitution_method_multiple.sh",
        }
    }
    
    make_dirs("{}".format(output_path))
    if update_mask:
        command = "sbatch {} {} {} {}/../.. jaccard".format(
            script_paths[mode]["distance_matrix_calculator"], config_file, prompts_file, 
            output_path
        )
        job_id = submit_slurm_job(command, "Jaccard distance matrix calculator")
        
        command = "sbatch --dependency=afterok:{} {} {} {} {}".format(
            job_id, script_paths[mode]["model_substitution_method"], config_file, 
            prompts_file, output_path
        )
        job_id = submit_slurm_job(command, "Modelling substitution method")
    else:
        command = "sbatch {} {} {} {}".format(
            script_paths[mode]["model_substitution_method"], config_file, prompts_file, 
            output_path
        )
        job_id = submit_slurm_job(command, "Modelling substitution method")
    
    command = "sbatch --dependency=afterok:{} {} {} {} {} dst".format(
        job_id, script_paths[mode]["distance_matrix_calculator"], config_file, 
        prompts_file, output_path
    )
    job_id = submit_slurm_job(command, "DST distance matrix calculator")
    return job_id

def slurm_run_recommender_system(
    job_id, config_file, prompts_file, output_path
):
    if job_id is None:
        command = "sbatch vasp.recommendation_system.sh {} {} {} {}/ers.pkl".format(
            config_file, prompts_file, 
            output_path, output_path
        )
    else:
        command = "sbatch --dependency=afterok:{} vasp.recommendation_system.sh {} {} {} {}/ers.pkl".format(
            job_id, config_file, prompts_file, 
            output_path, output_path
        )
    job_id = submit_slurm_job(command, "Recommending candidates")
    return job_id

def get_slurm_job_state(job_id):
    try:
        # Run the scontrol show job command
        result = subprocess.run(
            ["scontrol", "show", "job", str(job_id)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        # Parse the output to find the JobState field
        for line in result.stdout.splitlines():
            if "JobState" in line:
                # Extract and return the job state
                job_state = line.split('=')[1].split(' ')[0]
                return job_state

    except subprocess.CalledProcessError as e:
        print(f"Error running scontrol: {e.stderr}")
        return None

def wait_for_slurm_job_to_finish(job_id, check_interval=120):
    """
    Waits for a SLURM job to finish.

    Parameters:
    - job_id: The SLURM job ID to check.
    - check_interval: Time in seconds between checks.

    Returns:
    None
    """
    while True:
        # Use sacct to check job status, filtering for our job_id and getting the state
        cmd = ['sacct', '-j', str(job_id), '--format=State', '--noheader', '--parsable2']
        result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        
        # Extract the job state
        # job_state = result.stdout.strip().split('\n')[-1]

        job_state = get_slurm_job_state(job_id)
        print(job_state)
        
        # Check if the job state indicates completion
        if job_state in ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT']:
            print(f"----Job {job_id} finished with state: {job_state}----")
            break
        else:
            # print(f"Job {job_id} is still running with state: {job_state}. Checking again in {check_interval} seconds.")
            time.sleep(check_interval)
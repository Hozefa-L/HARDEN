import os
import subprocess

# Compile solidity smart contracts and collect runtime-bytecodes using solidity compiler solc
def sol_to_evm(sol_folder,evm_folder):
    for sol_name in os.listdir(sol_folder):
        if sol_name.endswith('.sol'):
            sol_path = os.path.join(sol_folder, sol_name)
            evm_path = os.path.join(evm_folder, sol_name)
            run_bytecodes = subprocess.run(["solc", "--bin-runtime", sol_path], capture_output=True,text=True) # Using -runtime for only runtime bytecodes, that is important information for detection

            evm = run_bytecodes.stdout
            error = run_bytecodes.stderr

            if evm:
                # evm_path = os.path.splitext(sol_path)[0] + ".evm"
                evm_path_new = os.path.splitext(evm_path)[0] + ".evm"
                with open(evm_path_new,'w') as evmp:
                    evmp.write(evm)
                    print(f"Bytecodes saved for {sol_name}")

            if error:
                print(f"Error occured for {sol_name}, change solc version")

# sol_folder_path = "Reentrancy_dataset/Messi-Q-reentrancy/sourcecode" #Messi-Q-reentrancy had SC without pragma & compiled with 0.4.19
# evm_folder_path = "Reentrancy_dataset/Messi-Q-reentrancy/bytecodes"
# sol_to_evm(sol_folder_path,evm_folder_path)

# Create flow graphs using Ethersolve with bytecode evm as input and digraph with (nodes,edges) as output"

def evm_to_cfg(evm_folder,html_folder):
    evm_files = [files for files in os.listdir(evm_folder) if files.endswith('.evm')]
    for file in evm_files:
        evm_path = os.path.join(evm_folder, file)
        dot_file = os.path.splitext(file)[0] + '.html' # Use .dot for a dot file
        cfg_path = os.path.join(html_folder, dot_file)
        subprocess.run(['java', '-jar', 'EtherSolve.jar', '-r', '-H', '-o', cfg_path, evm_path]) # -d for dot file and -H for html file

# evm_folder = "Reentrancy_dataset/Messi-Q-reentrancy/bytecode_collected"
# cfg_folder = "Reentrancy_dataset/cfg-reentrancy"
# html_folder = "Reentrancy_dataset/cfg-html"
# evm_to_cfg(evm_folder,html_folder)





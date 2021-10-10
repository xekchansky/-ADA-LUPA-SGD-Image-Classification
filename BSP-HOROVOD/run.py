import subprocess

debug = True

def get_names(num_workers=1):
    names = []
    for i in range(1, num_workers+1):
        names.append("worker" + str(i))
    return names

def launch_instances(names):

    commands = []
    for name in names:
            commands.append("multipass launch -d 25G -m 2G -n " + name + " --cloud-init cloud-config.yaml")
            commands.append("multipass mount /home/xekchansky/kursach/worker_dir " + name)

    for command in commands:
        if debug: print(command)
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        if debug: print(output.decode("utf-8") )
        if debug: print("Done")

def get_ips():
    if debug: print("getting ips")
    process = subprocess.Popen("multipass list".split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    outs = output.split()[4:]
    IP_dict = dict()
    for i in range(0, len(outs) // 6):
            IP_dict[outs[6*i].decode("utf-8")] = outs[6*i+2].decode("utf-8")
    return IP_dict

def add_known_hosts(names, ips):
    #this somehow doesn't work (enterprets >> as adress)

    if debug: print("adding ips to known hosts")
    command = "ssh-keyscan -t rsa "
    for name in names:
        command += ips[name] + ' '
    command += "> ~/.ssh/known_hosts"
    if debug: print(command)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    if debug: print(output.decode("utf-8"))

    if debug: print("checking ssh")
    command = "ssh " + str(ips[name]) + " ls"
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    if debug: print(output.decode("utf-8"))
    #assert output.decode("utf-8") == "kursach"
    if debug: print("Done")



def delete_instance(name):
    if debug: print("deleting instance: ", name)
    command = "multipass delete " + name
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    if debug: print(output.decode("utf-8") )

def delete_instances(names):
    if debug: print("deleting instances")
    for name in names:
        delete_instance(name)
    command = "multipass purge"
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    if debug: print(output.decode("utf-8") )

def get_libs(names, ips, CUDA=False, local=False):
    if debug: print("downloading libraries")
    for name in names:
        if debug: print(name)
        if debug: print(ips[name])

        #install CUDA
        if CUDA:
            if local:
                commands = ["wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin",
                            "sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600",
                            "wget https://developer.download.nvidia.com/compute/cuda/11.4.2/local_installers/cuda-repo-ubuntu2004-11-4-local_11.4.2-470.57.02-1_amd64.deb",
                            "sudo dpkg -i cuda-repo-ubuntu2004-11-4-local_11.4.2-470.57.02-1_amd64.deb",
                            "sudo apt-key add /var/cuda-repo-ubuntu2004-11-4-local/7fa2af80.pub",
                            "sudo apt-get update",
                            "sudo apt-get install cuda",
                            "cd /usr/local/cuda/samples/0_Simple/matrixMul",
                            "sudo make",
                            "cd",
                            "cd kursach/worker_dir",
                            "sudo dpkg -i libcudnn8_8.2.4.15-1+cuda11.4_amd64.deb"]
            else: #network
                commands = ["wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin",
                            "sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600",
                            "sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub",
                            'sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"',
                            "sudo apt-get update",
                            "sudo apt-get -y install cuda",
                            #"cd /usr/local/cuda/samples/0_Simple/matrixMul",
                            #"sudo make",
                            #"cd",
                            #"cd kursach/worker_dir",
                            "sudo dpkg -i kursach/worker_dir/libcudnn8_8.2.4.15-1+cuda11.4_amd64.deb"
                            ]
            for command in commands:
                if debug: print(command)
                process = subprocess.Popen("ssh " + str(ips[name]) + " " +command, stdout=subprocess.PIPE, shell=True)
                #process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()
                if debug: print(output.decode("utf-8"))

        #install tensorflow
        command = "ssh " + str(ips[name]) + " sudo pip install tensorflow"
        if CUDA: command += "-gpu"
        if debug: print(command)
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        if debug: print(output.decode("utf-8"))

        #install horovod
        command = "ssh " + ips[name] + " HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_MPI=1 HOROVOD_GPU=1" \
                                       " sudo pip install horovod"
        if debug: print(command)
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        if debug: print(output.decode("utf-8"))


def run_horovod(names, ips):
    # print horovod command
    command = "horovodrun -np " + str(len(names)) + " -H "
    for name in names:
        command += str(ips[name]) + ":1 "
    command += "python3 kursach/worker_dir/main.py"
    if debug: print(command)


delete = 0
names = get_names(1)

if delete:
    delete_instances(names)
else:
    launch_instances(names)
    ips = get_ips()
    add_known_hosts(names, ips)
    get_libs(names, ips, CUDA=True, local=False)
    run_horovod(names, ips)

#!/usr/bin/env python3
"""QMChargesTools: per-frame QM/MM ESP charges for FieldTools (experimental).

Splits a trajectory into restart files with cpptraj, lets sander (qm_theory=EXTERN)
write a Gaussian input with the MM point charges for each frame, runs Gaussian to
obtain CHelpG charges of the QM region and collects them into a file that
fieldtools reads with -use_qm_charges True.

Run `qmchargestools -h` for usage.
"""

import argparse
import datetime
import os
import re
import shlex
import shutil
import subprocess
import sys

TMP_MARKER = ".qmchargestools_tmp"   # Marks directories this script may clear

SBATCH_HEADER = """#!/bin/env bash
#SBATCH --job-name=Chrg
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:0
#SBATCH --mem=10000M

# 1. Load module(s)
module load  apps/amber18/tools19-packmol-mpi-gnu-7.2.0
module load apps/gaussian/16
"""


def out_stem(arg_out):
    return os.path.splitext(arg_out)[0]


def fkt_submit(arg_tmp_dir, arg_qm_submit):
    if arg_qm_submit == "False":
        return
    submission = os.path.join(arg_tmp_dir, "submission.in")
    with open(os.path.join(arg_tmp_dir, "submit.sh"), "w") as f:
        f.write(f"{arg_qm_submit} {shlex.quote(submission)}\n")
    print("QMMM calculation started with " + arg_qm_submit + " " + submission)
    with open(os.path.join(arg_tmp_dir, "submit.out"), "w") as f:
        subprocess.call(["bash", os.path.join(arg_tmp_dir, "submit.sh")], stdout=f)


def make_qm_dict(arg_out):
    with open(out_stem(arg_out) + "_qm_region.pdb") as f:
        pdb = [i.split() for i in f.readlines()]
    pdb = [i[2:5] for i in pdb if i and i[0] in ["ATOM", "HETATM"]]
    with open(out_stem(arg_out) + "_qm.dict", "w") as f:
        f.write("\n".join(["_".join(i) for i in pdb]))


def convert_charges_script(arg_out, arg_tmp_dir, n_frames):
    script = f"""# Read in gaussian logs and extract only the ESP charges into one file
import os
QM_Charges = []
for i in range(1, {n_frames + 1}):
    with open(os.path.join({arg_tmp_dir!r}, 'gaussian_' + str(i) + '.log')) as f:
        lines = f.readlines()
    i_start = None
    for out_i, line in enumerate(lines):
        if line.startswith(' ESP charges:'):
            i_start = out_i + 2
        if line.startswith(' Sum of ESP charges') and i_start is not None:
            QM_Charges += ['Frame ' + str(i) + '\\n'] + lines[i_start:out_i]
            i_start = None

with open({arg_out!r}, 'w') as f:
    f.writelines(QM_Charges)
"""
    with open(os.path.join(arg_tmp_dir, "convert_charges.py"), "w") as f:
        f.write(script)


def make_submission_script(arg_tmp_dir, arg_param, arg_rm_temp, arg_qm_submit, n_frames):
    q = shlex.quote
    submission = SBATCH_HEADER if arg_qm_submit == "sbatch" else "#!/bin/bash\n"
    submission += f"""
# 2. Set directories
cd {q(arg_tmp_dir)}
"""
    for i in range(1, n_frames + 1):
        submission += f"""
echo Frame {i} gaussian STARTED >> submission.out
sander -O -i amber.in -p {q(arg_param)} -o sander.out -c tmp.rst.{i} > sander_{i}.log 2>&1
cat  gaussian.in                      > gaussian_{i}.gjf
tail -n +5 gau_job.inp               >> gaussian_{i}.gjf
echo ""                              >> gaussian_{i}.gjf
g16  gaussian_{i}.gjf                 > gaussian_{i}.log
echo Frame {i} gaussian DONE >> submission.out
"""
    submission += f"""
python {q(os.path.join(arg_tmp_dir, "convert_charges.py"))}
"""
    if arg_rm_temp:
        submission += f"""
cd ..
rm -rf -- {q(arg_tmp_dir)}
"""
    with open(os.path.join(arg_tmp_dir, "submission.in"), "w") as f:
        f.write(submission)


def make_input_files(arg_tmp_dir, arg_qm_mask, arg_qm_charge, arg_qm_theory, arg_out, arg_param, arg_nc):
    ### Input file for sander
    amber = """Create Gaussian input file. Designed to crash sander after files are created
&cntrl
    ntpr = 1, ntwx = 0,
    imin = 1, maxcyc = 0,           !Single-point energy calculation
    ntb = 0,                        !Non-periodic
    cut = 9999.,                    !Calculate all interactions
    ifqnt = 1,                      !Switch on QM/MM coupled potential
/
&qmmm
    qmmask = '""" + arg_qm_mask + """',
    qmcharge = """ + arg_qm_charge + """,
    spin = 1,
    qm_theory = 'EXTERN',
    qmcut = 999.0,
    itrmax = 10000000,
    printcharges = 1,
&end
&gau
    method = 'XXX',
    basis = 'XXX',
&end
"""
    with open(os.path.join(arg_tmp_dir, "amber.in"), "w") as f:
        f.write(amber)

    ### Input header for gaussian
    gaussian = "%chk=" + os.path.join(arg_tmp_dir, "gaussian.chk") + """
%NProcShared=1
%mem=8192MB
# """ + arg_qm_theory + """ SCF=(Conver=8,verytight,Maxcyc=1000) Integral=Ultrafine NoSymm POP=CHelpG Charge
"""
    with open(os.path.join(arg_tmp_dir, "gaussian.in"), "w") as f:
        f.write(gaussian)

    cpptraj = "parm " + arg_param + """
trajin """ + arg_nc + """
autoimage
outtraj """ + os.path.join(arg_tmp_dir, "tmp.rst") + """ multi nobox
outtraj """ + os.path.join(arg_tmp_dir, "full.pdb") + """ onlyframes -1
strip !(""" + arg_qm_mask + """)
outtraj """ + out_stem(arg_out) + """_qm_region.pdb onlyframes -1
"""
    with open(os.path.join(arg_tmp_dir, "cpptraj.in"), "w") as f:
        f.write(cpptraj)


def count_rst_frames(arg_tmp_dir):
    frames = [f for f in os.listdir(arg_tmp_dir) if re.fullmatch(r"tmp\.rst\.\d+", f)]
    single = os.path.join(arg_tmp_dir, "tmp.rst")
    if not frames and os.path.isfile(single):
        # cpptraj does not number the output of a single-frame trajectory
        shutil.copy(single, single + ".1")
        return 1
    return len(frames)


def make_rst(arg_tmp_dir, arg_nc):
    with open(os.path.join(arg_tmp_dir, "cpptraj.out"), "w") as f:
        subprocess.run(["cpptraj", "-i", os.path.join(arg_tmp_dir, "cpptraj.in")], stdout=f)
    n_frames = count_rst_frames(arg_tmp_dir)
    if n_frames == 0:
        sys.exit("Error! cpptraj did not write any frames. See " + os.path.join(arg_tmp_dir, "cpptraj.out"))
    print(f"Cpptraj DONE. Trajectory {arg_nc} split into {n_frames} files {arg_tmp_dir}tmp.rst.X")
    return n_frames


def make_tmpdir(arg_tmp_dir):
    """Create the temporary directory, or clear one previously created by this script."""
    if os.path.isdir(arg_tmp_dir):
        contents = os.listdir(arg_tmp_dir)
        if contents and TMP_MARKER not in contents:
            sys.exit(f"Error! {arg_tmp_dir} is not empty and was not created by QMChargesTools. "
                     "Refusing to delete its contents; choose another -tmp_dir.")
        for f in contents:
            path = os.path.join(arg_tmp_dir, f)
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
    else:
        os.makedirs(arg_tmp_dir)
    open(os.path.join(arg_tmp_dir, TMP_MARKER), "w").close()


def load_qm_mask(arg_qm_mask):
    with open(arg_qm_mask) as f:
        return f.read().strip()


def str2bool(value):
    if value.lower() in ("true", "t", "yes", "y", "1"):
        return True
    if value.lower() in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"expected True or False, got '{value}'")


def inputParser(argv):
    parser = argparse.ArgumentParser(prog="qmchargestools", allow_abbrev=False,
                                     description="Calculate per-frame QM/MM ESP charges for FieldTools.")
    parser.add_argument("-nc", required=True, help="trajectory file")
    parser.add_argument("-parm", required=True, help="parameter file")
    parser.add_argument("-out", required=True, help="output file for the QM charges")
    parser.add_argument("-tmp_dir", default="tmp_chrg", help="temporary directory [default: %(default)s]")
    parser.add_argument("-qm_mask", required=True, help="file containing the QM region (amber selection mask)")
    parser.add_argument("-qm_charge", required=True, help="charge of the QM region")
    parser.add_argument("-qm_theory", default="M062X/6-31++g(d,p)", help="QM theory [default: %(default)s]")
    parser.add_argument("-submit", default="bash",
                        help="how the calculations are run: a submission command (sbatch, qsub, ...), "
                             "bash (run in the terminal) or False (only prepare the files) [default: %(default)s]")
    parser.add_argument("-rm_temp", type=str2bool, default=True,
                        help="remove the temporary directory when done [default: True]")
    args = parser.parse_args(argv)

    args.nc, args.parm, args.out, args.qm_mask = (os.path.abspath(p) for p in
                                                  (args.nc, args.parm, args.out, args.qm_mask))
    args.tmp_dir = os.path.join(os.path.abspath(args.tmp_dir), "")   # absolute, with trailing separator

    print("-nc        Trajectory file : ", args.nc)
    print("-parm      Parameter file  : ", args.parm)
    print("-out       Output file     : ", args.out)
    print("-tmp_dir   Temporary dir   : ", args.tmp_dir)
    print("-qm_mask   QM_mask         : ", args.qm_mask)
    print("-qm_charge QM_charge       : ", args.qm_charge)
    print("-qm_theory QM_theory       : ", args.qm_theory)
    print("-submit    Submit          : ", args.submit)
    print("-rm_temp   rm tmp data     : ", args.rm_temp)
    print()
    return args


def main(argv=None):
    args = inputParser(sys.argv[1:] if argv is None else argv)
    print("\nQM/MM calculation RUNNING : ", datetime.datetime.now())
    make_tmpdir(args.tmp_dir)
    qm_mask = load_qm_mask(args.qm_mask)
    make_input_files(args.tmp_dir, qm_mask, args.qm_charge, args.qm_theory, args.out, args.parm, args.nc)
    n_frames = make_rst(args.tmp_dir, args.nc)
    convert_charges_script(args.out, args.tmp_dir, n_frames)
    make_submission_script(args.tmp_dir, args.parm, args.rm_temp, args.submit, n_frames)
    make_qm_dict(args.out)
    fkt_submit(args.tmp_dir, args.submit)
    print("\nQM/MM calculation DONE using : ", args.submit, datetime.datetime.now())


if __name__ == "__main__":
    main()

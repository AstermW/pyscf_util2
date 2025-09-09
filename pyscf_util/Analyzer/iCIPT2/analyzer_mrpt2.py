from pyscf_util.Analyzer.iCIPT2.analyzer import *

############################################
## extract mrpt2/nevpt2 ##
############################################


def extract_mrenpt2_res(filename: str):
    with open(filename, "r") as file:
        content = file.read()
        if "--------------------- MRPT2 Driver End ---------------------" in content:
            matrix_elements = {}
            for line in content.splitlines():
                if re.match(r"\(\s*\d+,\s*\d+\s*\)\s*\|\s*-?\d+\.\d+", line):
                    indices, value = line.split("|")
                    i, j = map(int, re.findall(r"\d+", indices))
                    value = float(value.strip())
                    if not (
                        (i == 1 and j == 2)
                        or (i == 2 and j == 1)
                        or (i == 2 and j == 2)
                    ):
                        matrix_elements[(i, j)] = value
            return matrix_elements
        else:
            print(f"Error: File {filename} does not contain the required string.")


############################################
## extract nevpt2s ##
############################################

############################################
## case I one qmin extract both ept and etot
############################################


def extract_nevpt2s_old_type(filename: str):
    order = [
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (2, 0),
    ]
    with open(filename, "r") as file:
        content = file.read()
        if "--------------------- MRPT2 Driver End ---------------------" in content:
            pattern = r"perturbation.*\n\s*[^/]*/([^/]*)/"
            match = re.findall(pattern, content)
            assert len(match) == len(order)
            res = {}
            for i in range(len(order)):
                res[order[i]] = {"ept": float(match[i]), "etot": 0.0}

            # find etot #

            for line in content.splitlines():
                if re.match(r"\(\s*\d+,\s*\d+\s*\)\s*\|\s*-?\d+\.\d+", line):
                    indices, value = line.split("|")
                    i, j = map(int, re.findall(r"\d+", indices))
                    value = float(value.strip())
                    if not (
                        (i == 1 and j == 2)
                        or (i == 2 and j == 1)
                        or (i == 2 and j == 2)
                    ):
                        res[(i, j)]["etot"] = value

            return res

        else:
            print(f"Error: File {filename} does not contain the required string.")


def extract_nevpt2s_new_type(filename: str):
    with open(filename, "r") as file:
        content = file.read()
        if "--------------------- MRPT2 Driver End ---------------------" in content:

            res_tmp = []

            pattern = r"Qmin\s*=\s*((?:\d+\.\d+e-?\d+\s*)+)"
            matches = re.findall(pattern, content)
            matches = matches[0].split()
            matches = [float(match) for match in matches]

            # find etot #

            for line in content.splitlines():
                if re.match(r"\(\s*\d+,\s*\d+\s*\)\s*\|\s*-?\d+\.\d+", line):
                    indices, value = line.split("|")
                    i, j = map(int, re.findall(r"\d+", indices))
                    try:
                        value = float(value.strip())
                        if not (
                            (i == 1 and j == 2)
                            or (i == 2 and j == 1)
                            or (i == 2 and j == 2)
                        ):
                            # res[(i, j)]["etot"] = value
                            res_tmp.append([i, j, value])
                    except:
                        continue

            # print(res_tmp)

            assert len(res_tmp) == len(matches) * 10

            res = {}

            for idxqmin, qmin in enumerate(matches):
                res[qmin] = {}

                for i in range(idxqmin * 10, idxqmin * 10 + 5):
                    res[qmin][(res_tmp[i][0], res_tmp[i][1])] = {
                        "etot": 0.0,
                        "ept": res_tmp[i][2],
                    }

                for i in range(idxqmin * 10 + 5, idxqmin * 10 + 10):
                    res[qmin][(res_tmp[i][0], res_tmp[i][1])]["etot"] = res_tmp[i][2]

            return res

        else:
            print(f"Error: File {filename} does not contain the required string.")


############################################
## case II multi qmin
############################################


if __name__ == "__main__":

    filename = "mr_dyall.out"
    matrix_elements = extract_mrenpt2_res(filename)
    print(matrix_elements)

    filename = "mr_enpt2.out"
    matrix_elements = extract_mrenpt2_res(filename)
    print(matrix_elements)

    filename = "mr_sel.out1"
    nevpt2s = extract_nevpt2s_old_type(filename)
    print(nevpt2s)

    filename = "mr_sel.out2"
    nevpt2s = extract_nevpt2s_new_type(filename)
    print(nevpt2s)

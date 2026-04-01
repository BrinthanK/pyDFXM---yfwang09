import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np

import forward_model as fwd


_WORKER_STATE = {}


def build_scan_config():
    config = {}
    config["two_theta"] = 35.04
    config["hkl"] = [-1, 1, -1]
    config["Ug"] = np.array(
        [
            [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
            [1 / np.sqrt(2), 0, -1 / np.sqrt(2)],
            [-1 / np.sqrt(6), 2 / np.sqrt(6), -1 / np.sqrt(6)],
        ],
        dtype=float,
    )
    config["mu"] = np.deg2rad(1.95)
    config["psize"] = 40e-9
    config["zl_rms"] = 1.7e-6 / 2.35
    config["zl_truncation_sigma"] = 2.0
    config["Npixels"] = [600, 10, 200]
    config["Nrays"] = 5000000
    config["strain_profile_file"] = "strain_profile_3ns.csv"
    config["phi_start_deg"] = -0.03
    config["phi_stop_deg"] = 0.03
    config["phi_steps"] = 21
    config["num_workers"] = 10
    config["output_dir"] = "data"
    config["output_file"] = "strain_wave_phi_scan_3ns.h5"
    return config


def build_forward_dict(config):
    forward_dict = fwd.default_forward_dict()
    forward_dict["two_theta"] = config["two_theta"]
    forward_dict["hkl"] = config["hkl"]
    forward_dict["Ug"] = config["Ug"]
    forward_dict["mu"] = config["mu"]
    forward_dict["psize"] = config["psize"]
    forward_dict["zl_rms"] = config["zl_rms"]
    forward_dict["zl_truncation_sigma"] = config["zl_truncation_sigma"]
    forward_dict["Npixels"] = config["Npixels"]
    forward_dict["Nrays"] = config["Nrays"]
    return forward_dict


def build_phi_values(config):
    phi_start_rad = np.deg2rad(config["phi_start_deg"])
    phi_stop_rad = np.deg2rad(config["phi_stop_deg"])
    phi_values = np.linspace(phi_start_rad, phi_stop_rad, config["phi_steps"])
    return phi_values


def load_strain_profile(filepath):
    raw = np.genfromtxt(filepath, delimiter=",", names=True)
    xs_m = raw["distance_nm"] * 1e-9
    exx = raw["strain"]
    return xs_m, exx

    


def sort_strain_profile(xs_m, exx):
    needs_sort = np.any(np.diff(xs_m) < 0)
    if not needs_sort:
        return xs_m, exx
    sort_idx = np.argsort(xs_m)
    xs_sorted = xs_m[sort_idx]
    exx_sorted = exx[sort_idx]
    return xs_sorted, exx_sorted


def build_task_list(phi_values):
    tasks = []
    for j, phi in enumerate(phi_values):
        tasks.append((j, float(phi)))
    return tasks


def interpolate_strain(xs, profile_xs_m, profile_exx):
    eps_xx = np.interp(xs, profile_xs_m, profile_exx, left=0.0, right=0.0)
    return eps_xx


def build_sample_strain_tensor(eps_xx, shape):
    eps_sample = np.zeros(shape + (3, 3))
    eps_sample[..., 0, 0] = eps_xx
    return eps_sample


def rotate_strain_to_grain(eps_sample, Ug):
    eps_grain = np.einsum("ji,...jk,kl->...il", Ug, eps_sample, Ug)
    return eps_grain


def build_Fg_from_grain_strain(eps_grain, shape):
    Fg = np.zeros(shape + (3, 3))
    Fg[..., :, :] = np.eye(3)
    Fg += eps_grain
    return Fg


def make_Fg_func(Ug, profile_xs_m, profile_exx):
    def Fg_func(xg, yg, zg):
        rg = np.stack([xg, yg, zg], axis=-1)
        rs = np.einsum("ij,...j->...i", Ug, rg)
        xs = rs[..., 0]

        eps_xx = interpolate_strain(xs, profile_xs_m, profile_exx)
        eps_sample = build_sample_strain_tensor(eps_xx, xg.shape)
        eps_grain = rotate_strain_to_grain(eps_sample, Ug)
        Fg = build_Fg_from_grain_strain(eps_grain, xg.shape)
        return Fg

    return Fg_func


def init_compute_worker(worker_config):
    forward_dict = worker_config["forward_dict"]
    res_fn_file = worker_config["res_fn_file"]
    Ug = worker_config["Ug"]
    profile_xs_m = worker_config["profile_xs_m"]
    profile_exx = worker_config["profile_exx"]

    model = fwd.DFXM_forward(forward_dict, load_res_fn=res_fn_file)
    Fg_func = make_Fg_func(Ug, profile_xs_m, profile_exx)

    _WORKER_STATE["model"] = model
    _WORKER_STATE["Fg_func"] = Fg_func


def compute_one_task(task):
    j, phi = task
    model = _WORKER_STATE["model"]
    Fg_func = _WORKER_STATE["Fg_func"]

    model.d["phi"] = phi
    im, _, rulers = model.forward(Fg_func, timeit=False)

    result = {}
    result["kind"] = "result"
    result["j"] = j
    result["phi"] = phi
    result["im"] = im
    result["imax"] = float(np.max(im))
    result["imin"] = float(np.min(im))
    result["rulers"] = rulers
    return result


def writer_worker(queue, writer_config):
    total_tasks = writer_config["total_tasks"]
    phi_values = writer_config["phi_values"]
    saved_h5_file = writer_config["saved_h5_file"]
    forward_dict = writer_config["forward_dict"]
    config = writer_config["config"]
    profile_xs_m = writer_config["profile_xs_m"]
    profile_exx = writer_config["profile_exx"]

    os.makedirs(os.path.dirname(saved_h5_file), exist_ok=True)

    done_count = 0
    dset_images = None

    with h5py.File(saved_h5_file, "w") as hf:
        scan_grp = hf.create_group("phi_scan")
        scan_grp.create_dataset("phi_values", data=phi_values)

        dset_imax = scan_grp.create_dataset(
            "Imax", shape=(len(phi_values),), dtype=np.float64
        )
        dset_imin = scan_grp.create_dataset(
            "Imin", shape=(len(phi_values),), dtype=np.float64
        )

        rulers_saved = None

        while done_count < total_tasks:
            msg = queue.get()

            if msg["kind"] != "result":
                done_count += 1
                print("Writer received non-result payload", flush=True)
                continue

            j = msg["j"]
            im = msg["im"]

            if dset_images is None:
                image_shape = im.shape
                full_shape = (len(phi_values),) + image_shape
                dset_images = scan_grp.create_dataset(
                    "images", shape=full_shape, dtype=im.dtype, compression="gzip"
                )

            dset_images[j] = im
            dset_imax[j] = msg["imax"]
            dset_imin[j] = msg["imin"]

            if rulers_saved is None:
                rulers_saved = msg["rulers"]

            done_count += 1
            if done_count % 5 == 0 or done_count == total_tasks:
                print(
                    "Writer progress: %d / %d" % (done_count, total_tasks), flush=True
                )

        if rulers_saved is not None:
            scan_grp.create_dataset("ruler_xl", data=rulers_saved[0])
            scan_grp.create_dataset("ruler_yl", data=rulers_saved[1])
            scan_grp.create_dataset("ruler_zl", data=rulers_saved[2])

        strain_grp = hf.create_group("strain_profile")
        strain_grp.create_dataset("xs_m", data=profile_xs_m)
        strain_grp.create_dataset("exx", data=profile_exx)

        fwd_grp = hf.create_group("forward_model")
        for key, val in forward_dict.items():
            if isinstance(val, np.ndarray):
                fwd_grp.create_dataset(key, data=val)
            else:
                fwd_grp.attrs[key] = val

        hf.attrs["two_theta"] = config["two_theta"]
        hf.attrs["mu_rad"] = config["mu"]
        hf.attrs["psize"] = config["psize"]
        hf.attrs["strain_profile_file"] = config["strain_profile_file"]
        hf.attrs["num_workers"] = config["num_workers"]
        hf.attrs["phi_start_deg"] = config["phi_start_deg"]
        hf.attrs["phi_stop_deg"] = config["phi_stop_deg"]
        hf.attrs["phi_steps"] = config["phi_steps"]

        hf.create_dataset("images", data=scan_grp["images"], compression="gzip")
        hf.create_dataset("phi_values", data=phi_values)
        hf.create_dataset("Imax", data=scan_grp["Imax"])
        hf.create_dataset("Imin", data=scan_grp["Imin"])

    print("Writer finished: %s" % saved_h5_file, flush=True)


def run_parallel_scan(config):
    forward_dict = build_forward_dict(config)
    phi_values = build_phi_values(config)

    xs_raw, exx_raw = load_strain_profile(config["strain_profile_file"])
    profile_xs_m, profile_exx = sort_strain_profile(xs_raw, exx_raw)

    datapath = config["output_dir"]
    os.makedirs(datapath, exist_ok=True)
    res_fn_file = os.path.join(datapath, "Res_qi_strain_wave.npz")
    saved_h5_file = os.path.join(datapath, config["output_file"])

    tasks = build_task_list(phi_values)
    total_tasks = len(tasks)

    writer_config = {}
    writer_config["total_tasks"] = total_tasks
    writer_config["phi_values"] = phi_values
    writer_config["saved_h5_file"] = saved_h5_file
    writer_config["forward_dict"] = forward_dict
    writer_config["config"] = config
    writer_config["profile_xs_m"] = profile_xs_m
    writer_config["profile_exx"] = profile_exx

    worker_config = {}
    worker_config["forward_dict"] = forward_dict
    worker_config["res_fn_file"] = res_fn_file
    worker_config["Ug"] = config["Ug"]
    worker_config["profile_xs_m"] = profile_xs_m
    worker_config["profile_exx"] = profile_exx

    queue = mp.Queue(maxsize=2 * config["num_workers"])
    writer_process = mp.Process(target=writer_worker, args=(queue, writer_config))
    writer_process.start()

    start_time = time.time()
    finished = 0
    print(
        "Starting parallel phi scan with %d workers for %d tasks"
        % (config["num_workers"], total_tasks),
        flush=True,
    )

    with ProcessPoolExecutor(
        max_workers=config["num_workers"],
        initializer=init_compute_worker,
        initargs=(worker_config,),
    ) as executor:
        futures = [executor.submit(compute_one_task, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            queue.put(result)
            finished += 1
            if finished % 5 == 0 or finished == total_tasks:
                elapsed = time.time() - start_time
                print(
                    "Compute progress: %d / %d (%.1fs)"
                    % (finished, total_tasks, elapsed),
                    flush=True,
                )

    writer_process.join()
    elapsed_total = time.time() - start_time
    print("All done in %.1fs" % elapsed_total, flush=True)
    print("Output file: %s" % saved_h5_file, flush=True)


def main():
    config = build_scan_config()
    run_parallel_scan(config)


if __name__ == "__main__":
    main()

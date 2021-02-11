import os
import numpy as np

            
def get_waves(v, T1=0.20, T2=0.10, verbose=False, min_wave_size=5):
    """Annotate waves on prices.
    Args:
        v : The prices.
        T1 : The threshold for starting a type of trend.
        T2 : The threshold for maintaining a type of trend.
        verbose: Show information.
    Returns:
        A list of 5-tuples, (wave start idx, wave start value, wave end idx,
        wave end value, wave type). wave type: -1, 0, 1 means decreasing, 
        null, and increasing waves, respectively.
    """
    waves = []
    last_wave_idx = 0
    wave_type = 0
    lmax, lmin = v[0], v[0]
    lmaxi, lmini = 0, 0
    pinc_rate = 0
    pdec_rate = 0

    def _rec_wave(x1, y1, x2, y2, t):
        waves.append([x1, y1, x2, y2, t])

    for i in range(1, v.shape[0]):
        if lmax < v[i]:
            lmaxi, lmax = i, v[i]
        if lmin > v[i]:
            lmini, lmin = i, v[i]

        inc_rate = v[i] / lmin - 1
        dec_rate = 1 - v[i] / lmax
            
        if inc_rate >= T1 and wave_type != 1:
            if verbose:
                print(f"=> Start inc {lmin:.2f}({lmini}) -> {v[i]:.2f}({i})")
            _rec_wave(last_wave_idx, v[last_wave_idx],
                        lmini - 1, v[lmini - 1], wave_type)
            last_wave_idx = lmini
            lmaxi, lmax = i, v[i]
            wave_type = 1
        elif wave_type == -1 and inc_rate >= T2:
            if verbose:
                print(f"=> dec -> null, {lmin:.2f}({lmini}) -> {v[i]:.2f}({i})")
            _rec_wave(last_wave_idx, v[last_wave_idx],
                        lmini - 1, v[lmini - 1], wave_type)
            last_wave_idx = lmini
            lmaxi, lmax = i, v[i]
            wave_type = 0
        if dec_rate >= T1 and wave_type != -1:
            if verbose:
                print(f"=> Start dec, {lmax:.2f}({lmaxi}) -> {v[i]:.2f}({i})")
            _rec_wave(last_wave_idx, v[last_wave_idx],
                        lmaxi - 1, v[lmaxi - 1], wave_type)
            last_wave_idx = lmaxi
            lmini, lmin = i, v[i]
            wave_type = -1
        elif wave_type == 1 and dec_rate >= T2:
            if verbose:
                print(f"=> inc -> null, {lmax:.2f}({lmaxi}) -> {v[i]:.2f}({i})")
            # encode last wave
            _rec_wave(last_wave_idx, v[last_wave_idx],
                        lmaxi - 1, v[lmaxi - 1], wave_type)
            last_wave_idx = lmaxi
            lmini, lmin = i, v[i]
            wave_type = 0

        if inc_rate >= T2 and pinc_rate < T2:
            lmaxi, lmax = i, v[i]
        if dec_rate >= T2 and pdec_rate < T2:
            lmini, lmin = i, v[i]

        pinc_rate = v[i] / lmin - 1
        pdec_rate = 1 - v[i] / lmax

    _rec_wave(last_wave_idx, v[last_wave_idx], v.shape[0] - 1, v[-1], wave_type)
    waves = [w for w in waves if w[2] - w[0] > min_wave_size]
    return waves


def _label_wave(v, waves, infos, x1, y1, x2, y2, t, T1, T2):
    # current maximum, minimum and their locations
    cmax, cmaxi, cmin, cmini = v[x2], x2, v[x2], x2
    # maximum increase and their start and end
    mi, mist, mied = 0, x2, x2
    # maximum decrease and their start and end
    md, mdst, mded = 0, x2, x2
    for i in range(x2, -1, -1):
        if cmax < v[i]:
            cmaxi, cmax = i, v[i]
        if cmin > v[i]:
            cmini, cmin = i, v[i]
        cinc, cdec = cmax / v[i] - 1, cmin / v[i] - 1
        if cinc > mi:
            mi, mist, mied = cinc, i, cmaxi
        if cdec < md:
            md, mdst, mded = cdec, i, cmini
        # whether should terminate
        if t == 1 and -md > T2: 
            x1 = i + 1
            break # inc wave, check retract
        elif t == 0 and (mi > T1 or -md > T1):
            x1 = i + 1
            break # null wave, check wave type
        elif t == -1 and (mi > T2):
            x1 = i + 1
            break # dec wave, check increase
        
        #infos[i] = cinc, mi, mist, mied, cdec, md, mdst, mded
        infos[i] = cinc, cdec
        if i == 0:
            x1 = 0
    x1, x2 = max(0, x1), max(0, x2)
    y1, y2 = v[x1], v[x2]
    waves.append([x1, y1, x2, y2, t])


def merge_waves(waves):
    """Merge waves with same type."""
    i = 1
    waves.sort(key=lambda x: x[0])
    while i < len(waves):
        pw = waves[i - 1]
        cw = waves[i]
        if pw[-1] == cw[-1]: # merge same waves
            waves[i] = [pw[0], pw[1], cw[2], cw[3], cw[-1]]
            del waves[i - 1] # this wave is merged into previous one
            continue
        i += 1
        

def post_process_waves(waves, v, min_wave_size):
    """Remove unncesssary null waves."""
    i = 1
    while i < len(waves):
        pw = waves[i - 1]
        cw = waves[i]
        #if pw[-1] == cw[-1]: # merge same waves
        #    waves[i] = [pw[0], pw[1], cw[2], cw[3], cw[-1]]
        #    del waves[i - 1] # this wave is merged into previous one
        #    continue
        
        # resolve overlap
        if cw[-1] == 0 and pw[-1] != 0: 
            waves[i] = [pw[2] + 1, v[pw[2] + 1], cw[2], cw[3], cw[-1]]
            if waves[i][2] - waves[i][0] < min_wave_size:
                del waves[i] # small wave: delete
                continue
        elif cw[-1] != 0 and pw[-1] == 0:
            waves[i - 1] = [pw[0], pw[1], cw[0] - 1, v[cw[0] - 1], pw[-1]]
            if waves[i - 1][2] - waves[i - 1][0] < min_wave_size:
                del waves[i - 1] # small wave: delete
                continue
        
        i += 1


def label_waves(v, T1=0.20, T2=0.10, verbose=False, min_wave_size=5):
    """Annotate waves on prices as point-wise label.
    Args:
        v : The prices.
        T1 : The threshold for starting a type of trend.
        T2 : The threshold for maintaining a type of trend.
        verbose: Show information.
    Returns:
        A list of 5-tuples, (wave start idx, wave start value, wave end idx,
        wave end value, wave type). wave type: -1, 0, 1 means decreasing, 
        null, and increasing waves, respectively.
    """
    waves = []
    # 8: wave profit, largest increase, start, end;
    # wave retract, largest decrease, start, end
    # 2: wave profit, wave retract
    infos = np.zeros((v.shape[0], 2))# 8)) 
    wave_idx = 0
    wave_type = 0
    lmax, lmin = v[0], v[0]
    lmaxi, lmini = 0, 0
    pinc_rate = 0
    pdec_rate = 0

    for i in range(1, v.shape[0]):
        if lmax < v[i]:
            lmaxi, lmax = i, v[i]
        if lmin > v[i]:
            lmini, lmin = i, v[i]

        inc_rate = v[i] / lmin - 1
        dec_rate = 1 - v[i] / lmax
            
        if inc_rate >= T1 and wave_type != 1:
            _label_wave(v, waves, infos, wave_idx, v[wave_idx],
                        lmini - 1, v[lmini - 1], wave_type, T1, T2)
            wave_idx = lmini
            lmaxi, lmax = i, v[i]
            wave_type = 1
        elif wave_type == -1 and inc_rate >= T2:
            _label_wave(v, waves, infos, wave_idx, v[wave_idx],
                        lmini - 1, v[lmini - 1], wave_type, T1, T2)
            wave_idx = lmini
            lmaxi, lmax = i, v[i]
            wave_type = 0
        elif dec_rate >= T1 and wave_type != -1:
            _label_wave(v, waves, infos, wave_idx, v[wave_idx],
                        lmaxi - 1, v[lmaxi - 1], wave_type, T1, T2)
            wave_idx = lmaxi
            lmini, lmin = i, v[i]
            wave_type = -1
        elif wave_type == 1 and dec_rate >= T2:
            _label_wave(v, waves, infos, wave_idx, v[wave_idx],
                        lmaxi - 1, v[lmaxi - 1], wave_type, T1, T2)
            wave_idx = lmaxi
            lmini, lmin = i, v[i]
            wave_type = 0

        if inc_rate >= T2 and pinc_rate < T2:
            lmaxi, lmax = i, v[i]
        if dec_rate >= T2 and pdec_rate < T2:
            lmini, lmin = i, v[i]

        pinc_rate = v[i] / lmin - 1
        pdec_rate = 1 - v[i] / lmax

    _label_wave(v, waves, infos, wave_idx, v[wave_idx],
        v.shape[0] - 1, v[-1], wave_type, T1, T2)
    post_process_waves(waves, v, min_wave_size)
    return waves, infos
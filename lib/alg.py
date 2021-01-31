import os

            
def get_waves(v, T1=0.2, T2=0.10, verbose=False):
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

        if inc_rate > T2 and pinc_rate < T2:
            lmaxi, lmax = i, v[i]
        if dec_rate > T2 and pdec_rate < T2:
            lmini, lmin = i, v[i]
            
        if inc_rate > T1 and wave_type != 1:
            if verbose:
                print(f"=> Start inc {lmin:.2f}({lmini}) -> {v[i]:.2f}({i})")
            _rec_wave(last_wave_idx, v[last_wave_idx],
                        lmini, lmin, wave_type)
            last_wave_idx = lmini
            wave_type = 1
        elif wave_type == -1 and inc_rate > T2:
            if verbose:
                print(f"=> dec -> null, {lmin:.2f}({lmini}) -> {v[i]:.2f}({i})")
            _rec_wave(last_wave_idx, v[last_wave_idx],
                        lmini, lmin, wave_type)
            last_wave_idx = lmini
            wave_type = 0

        if dec_rate > T1 and wave_type != -1:
            if verbose:
                print(f"=> Start dec, {lmax:.2f}({lmaxi}) -> {v[i]:.2f}({i})")
            _rec_wave(last_wave_idx, v[last_wave_idx],
                        lmaxi, lmax, wave_type)
            last_wave_idx = lmaxi
            wave_type = -1
        elif wave_type == 1 and dec_rate > T2:
            if verbose:
                print(f"=> inc -> null, {lmax:.2f}({lmaxi}) -> {v[i]:.2f}({i})")
            # encode last wave
            _rec_wave(last_wave_idx, v[last_wave_idx],
                        lmaxi, lmax, wave_type)
            last_wave_idx = lmaxi
            wave_type = 0

        pinc_rate = v[i] / lmin - 1
        pdec_rate = 1 - v[i] / lmax

    _rec_wave(last_wave_idx, v[last_wave_idx], v.shape[0] - 1, v[-1], wave_type)
    return waves
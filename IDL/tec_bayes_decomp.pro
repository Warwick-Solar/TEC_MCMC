;+
; NAME:
;   TEC_BAYES_DECOMP
;
; PURPOSE:
;   Bayesian background-flare decomposition of GNSS total electron content
;   (TEC) curves. Each curve (one satellite pass) is fitted over its whole
;   duration as a smooth background plus K flare-induced peaks:
;
;     y(t)   = B(t) + sum_k F_k(t)
;     B(t)   = c0 + sum_{j=1..4} A_j sin(2 pi t / P_j + phi_j)
;     F_k(t) = I_k exp[ (t - t_k)/tau_r,k]   for t <  t_k
;              I_k exp[-(t - t_k)/tau_d,k]   for t >= t_k
;
;   The posterior is sampled with the Metropolis-Hastings MCMC of SoBAT
;   (Anfinogentov et al. 2021, ApJS 252, 11), with the noise standard
;   deviation as an extra free parameter. K and the initial t_k are set by
;   reference (solar) peak times that fall within the curve. Curves with
;   fewer than 3 reference peaks get one extra "dummy" peak at a random
;   time, used to estimate the level of spurious amplitudes (noise test).
;   The TEC response (Delta TEC) is y(t) - B(t).
;
; CALLING SEQUENCE:
;   tec_bayes_decomp, csv_file, t_peaks_solar [, OUT_DIR=, CURVES=, $
;     BURN_IN=, N_SAMPLES=, N_PPD=, CONF_LEVEL=, MIN_POINTS=, SEED=, $
;     YRANGE_DT=, /NOPLOT]
;
; INPUTS:
;   csv_file      - comma-separated file, no header. Column 1: time [s, UT
;                   seconds of day]; columns 2..N+1: TEC [TECU] of N
;                   satellite passes, NaN where a pass has no data.
;   t_peaks_solar - reference peak times [min, UT minutes of day], e.g.
;                   peaks of the He II 30.4 nm / Fe XV 28.4 nm emission.
;
; KEYWORDS:
;   OUT_DIR    - output directory (default: directory of csv_file).
;   CURVES     - 0-based indices of the TEC columns to process
;                (0 = CSV column 2; default: all).
;   BURN_IN    - MCMC burn-in samples (default 1e5).
;   N_SAMPLES  - MCMC posterior samples (default 1e5).
;   N_PPD      - posterior-predictive samples used for the credible band
;                of the fit (default 2e4).
;   CONF_LEVEL - credible level (default 0.95).
;   MIN_POINTS - curves with fewer valid points are skipped (default 30).
;   SEED       - seed for the dummy-peak position (default: system clock).
;                MCMC sampling in SoBAT is always clock-seeded.
;   YRANGE_DT  - y-range of the summary Delta TEC plot (default [-1, 5]).
;   NOPLOT     - set to skip all graphics and PNG output.
;
; OUTPUTS:
;   <OUT_DIR>/<name>_bayes.sav, where <name> is csv_file without '.csv'.
;   Lists, one element per processed curve (time in UT minutes of day):
;     time_list, obs_list   - time and observed TEC
;     fits_list             - best-fit (maximum a posteriori) model
;     trend_list            - background B(t)
;     flares_list           - sum of peaks F_k(t), incl. any dummy peak
;     cred_up_list,
;     cred_down_list        - credible band of the posterior predictive
;     params_list           - best-fit parameters, see PARAMETER LAYOUT
;     param_ci_list         - [n_par, 2] credible intervals of params_list
;     maxpos_list           - indices of the initial peak times in
;                             time_list[i] (dummy, if any, is last)
;     n_ref_list            - number of reference peaks within the curve
;     column_list           - curve index in the CSV (as in CURVES)
;   Also saved: t_peaks_solar, dir (OUT_DIR), filename (<name>).
;   PNG figures (unless /NOPLOT): <name>_<col>_init.png (initial guess),
;   <name>_<col>_fit.png (fit), <name>_all.png, <name>_all_detrended.png.
;
; PARAMETER LAYOUT (params_list[i]):
;   [0]          c0
;   [1+3(j-1) : 3+3(j-1)]   A_j, P_j [min], phi_j  for harmonics j = 1..4
;   [13+4k : 16+4k]         I_k, t_k, tau_r,k, tau_d,k  for peaks k = 0..
;   [-1]         noise standard deviation
;   t_k [min] is counted from the first time point of the curve, i.e.
;   UT minute = t_k + time_list[i][0]. The last peak block is the dummy
;   peak; for curves with >= 3 reference peaks (no dummy) it is NaN, so
;   the dummy is always params[-5:-2].
;
; DEPENDENCIES:
;   SoBAT  (https://github.com/Sergey-Anfinogentov/SoBAT)
;   Coyote IDL library (https://github.com/idl-coyote/coyote), for
;   cgPercentiles, required by SoBAT.
;
; EXAMPLE:
;   tec_bayes_decomp, 'tec_20170906.csv', [544., 549., 555., 561.], $
;     OUT_DIR='results/'
;
; REFERENCE:
;   Bekker, S. Z., Kolotkov, D. Y., & Belov, S. A., A Bayesian approach to
;   GNSS TEC detrending during solar flares: substructure and relaxation of
;   the ionospheric response, Geophys. Res. Lett. (submitted).
;
; AUTHOR:
;   Dmitrii Y. Kolotkov, University of Warwick, d.kolotkov.1@warwick.ac.uk
;
; MODIFICATION HISTORY:
;   2026-10-15  Public release.
;-

;------------------------------------------------------------------------------
; Model: background (offset + 4 harmonics) + K asymmetric exponential peaks.
; K = (n_elements(params) - 13)/4, so a trailing noise parameter is ignored.
;------------------------------------------------------------------------------
FUNCTION trend_harmonic_flares, t, params, _extra = _extra
  compile_opt idl2

  y = params[0] + dblarr(n_elements(t))
  FOR j = 0, 3 DO BEGIN
    p = params[1 + 3*j : 3 + 3*j]                ; amplitude, period, phase
    y += p[0]*sin((2d*!dpi/p[1])*t + p[2])
  ENDFOR

  n_flares = (n_elements(params) - 13)/4
  FOR k = 0, n_flares - 1 DO BEGIN
    p = params[13 + 4*k : 16 + 4*k]              ; I, t_peak, tau_rise, tau_decay
    dt = t - p[1]
    ir = where(dt LT 0, nr, complement = id, ncomplement = nd)
    IF nr GT 0 THEN y[ir] += p[0]*exp( dt[ir]/p[2])
    IF nd GT 0 THEN y[id] += p[0]*exp(-dt[id]/p[3])
  ENDFOR

  RETURN, y
END

;------------------------------------------------------------------------------
; Fast reader for a header-less numeric CSV (NaN allowed). Returns
; dblarr(n_columns, n_rows). Much faster than READ_CSV for 1000s of columns.
;------------------------------------------------------------------------------
FUNCTION tbd_read_csv, file
  compile_opt idl2

  IF ~file_test(file, /read) THEN message, 'Cannot read file: ' + file
  lines = strarr(file_lines(file))
  openr, lun, file, /get_lun
  readf, lun, lines
  free_lun, lun
  lines = lines[where(strlen(strtrim(lines, 2)) GT 0, n_rows)]

  n_cols = n_elements(strsplit(lines[0], ',', /preserve_null))
  data = dblarr(n_cols, n_rows)
  FOR r = 0L, n_rows - 1 DO BEGIN
    tok = strsplit(lines[r], ',', /extract, /preserve_null)
    IF n_elements(tok) NE n_cols THEN $
      message, 'Row ' + strtrim(r + 1, 2) + ': wrong number of columns'
    e = where(strtrim(tok, 2) EQ '', ne_)
    IF ne_ GT 0 THEN tok[e] = 'NaN'
    data[*, r] = double(tok)
  ENDFOR

  RETURN, data
END

;------------------------------------------------------------------------------
; Remove step-like artefacts: each run of identical consecutive values is
; replaced by its midpoint, then linearly re-interpolated onto t.
;------------------------------------------------------------------------------
FUNCTION tbd_destep, t, y
  compile_opt idl2

  n = n_elements(y)
  IF n LT 3 THEN RETURN, y
  brk = where(y[1:*] NE y[0:-2], nb)             ; last index of each run
  IF nb EQ 0 THEN RETURN, y
  mid = ([0L, brk + 1] + [brk, n - 1])/2
  RETURN, interpol(y[mid], t[mid], t)
END

;------------------------------------------------------------------------------
; Indices of the samples of t lying within half a time step of t_peaks.
; Returns -1 if none.
;------------------------------------------------------------------------------
FUNCTION tbd_match_peaks, t, t_peaks
  compile_opt idl2

  dt = t[1] - t[0]
  pos = -1L
  FOREACH tp, t_peaks DO BEGIN
    ind = where(abs(t - tp) LE 0.5*dt, count)
    IF count GT 0 THEN pos = [pos, ind]
  ENDFOREACH
  RETURN, n_elements(pos) EQ 1 ? -1L : pos[1:*]
END

;------------------------------------------------------------------------------
; Random index for the dummy peak in a curve of n points (margin = n/10):
;   no reference peaks - anywhere in the curve;
;   one                - in the opposite half, away from the edges;
;   two                - as far as possible from both, away from the edges.
;------------------------------------------------------------------------------
FUNCTION tbd_dummy_position, ref_pos, n, seed
  compile_opt idl2

  margin = n/10L

  IF ref_pos[0] EQ -1 THEN RETURN, long(randomu(seed)*n)

  IF n_elements(ref_pos) EQ 1 THEN BEGIN
    IF ref_pos[0] LT n/2 THEN BEGIN
      lo = max([margin, n/2])
      hi = n - margin - 1
    ENDIF ELSE BEGIN
      lo = margin
      hi = min([n/2, n - margin]) - 1
    ENDELSE
    cand = lindgen(hi - lo + 1) + lo
  ENDIF ELSE BEGIN
    valid = lindgen(n - 2*margin) + margin
    mindist = replicate(n, n_elements(valid))
    FOREACH p, ref_pos DO mindist = mindist < abs(valid - p)
    cand = valid[where(mindist EQ max(mindist))]
  ENDELSE

  RETURN, cand[long(randomu(seed)*n_elements(cand))]
END

;------------------------------------------------------------------------------
; Open a graphics window only if the current device supports windows
; (allows headless use with SET_PLOT, 'Z').
;------------------------------------------------------------------------------
PRO tbd_window, id, _extra = _extra
  compile_opt idl2
  IF (!d.flags AND 256) NE 0 THEN window, id, _extra = _extra
END

;------------------------------------------------------------------------------
; Main procedure
;------------------------------------------------------------------------------
PRO tec_bayes_decomp, csv_file, t_peaks_solar, OUT_DIR = out_dir, $
  CURVES = curves, BURN_IN = burn_in, N_SAMPLES = n_samples, N_PPD = n_ppd, $
  CONF_LEVEL = conf_level, MIN_POINTS = min_points, SEED = seed, $
  YRANGE_DT = yrange_dt, NOPLOT = noplot

  compile_opt idl2

  IF n_params() LT 2 THEN $
    message, 'Usage: tec_bayes_decomp, csv_file, t_peaks_solar [, keywords]'

  ; ---------------- settings ----------------
  IF n_elements(burn_in)    EQ 0 THEN burn_in    = 100000L
  IF n_elements(n_samples)  EQ 0 THEN n_samples  = 100000L
  IF n_elements(n_ppd)      EQ 0 THEN n_ppd      = 20000L
  IF n_elements(conf_level) EQ 0 THEN conf_level = 0.95d
  IF n_elements(min_points) EQ 0 THEN min_points = 30L
  IF n_elements(yrange_dt)  EQ 0 THEN yrange_dt  = [-1d, 5d]
  rseed = n_elements(seed) GT 0 ? seed : systime(1)
  n_ppd = n_ppd < n_samples
  do_plot = ~keyword_set(noplot)

  filename = file_basename(csv_file, '.csv')
  dir = n_elements(out_dir) GT 0 ? out_dir : file_dirname(csv_file)
  IF ~file_test(dir, /directory) THEN file_mkdir, dir
  IF strmid(dir, 0, /reverse_offset) NE path_sep() THEN dir += path_sep()

  ; Prior ranges: offset, harmonic amplitude [TECU], harmonic period [min]
  ; and, in units of the curve noise level, flare amplitude; flare peak time
  ; relative to the reference time, rise and decay times [min].
  c0_range  = [-5d2, 5d2]
  amp_range = [0d, 5d2]
  per_range = [50d, 24d*60d]
  fl_amp    = 20d
  fl_tpeak  = [-1d, 10d]
  fl_rise   = [0d, 5d]
  fl_decay  = [2d, 100d]

  ; ---------------- read data ----------------
  print, 'Reading ' + csv_file + ' ...'
  data = tbd_read_csv(csv_file)
  t_full = reform(data[0, *])/60d                ; s -> min
  n_obs = (size(data, /dimensions))[0] - 1
  IF n_elements(curves) EQ 0 THEN curves = lindgen(n_obs)
  IF max(curves) GE n_obs OR min(curves) LT 0 THEN $
    message, 'CURVES must be within 0..' + strtrim(n_obs - 1, 2)

  time_list = list() & obs_list = list() & column_list = list()
  FOREACH ic, curves DO BEGIN
    y = reform(data[ic + 1, *])
    ind = where(finite(y), count)
    IF count LT min_points THEN BEGIN
      print, 'Curve ' + strtrim(ic, 2) + ' skipped: ' + strtrim(count, 2) + ' points'
      CONTINUE
    ENDIF
    time_list.add, t_full[ind]
    obs_list.add, y[ind]
    column_list.add, ic
  ENDFOREACH
  data = 0                                       ; free memory
  n_curves = n_elements(time_list)
  IF n_curves EQ 0 THEN message, 'No curves to process.'
  print, strtrim(n_curves, 2) + ' curves to process.'

  ; global plot ranges
  time_min = !values.d_infinity & time_max = -time_min
  obs_min  = time_min           & obs_max  = time_max
  FOR i = 0, n_curves - 1 DO BEGIN
    time_min = time_min < min(time_list[i], max = m) & time_max = time_max > m
    obs_min  = obs_min  < min(obs_list[i],  max = m) & obs_max  = obs_max  > m
  ENDFOR
  yrange_all = [obs_min - 0.2*abs(obs_min), obs_max + 0.2*abs(obs_max)]

  IF do_plot THEN BEGIN
    device, get_decomposed = decomposed0
    device, decomposed = 0
    loadct, 39, /silent                          ; 0 black, 60 blue, 150 green, 250 red, 255 white
    tbd_window, 0
    plot, [0], [0], /nodata, xrange = [time_min, time_max], yrange = yrange_all, $
      /xstyle, /ystyle, color = 0, background = 255, $
      xtitle = 'Time [UT min]', ytitle = 'TEC [TECU]', title = filename
    FOR i = 0, n_curves - 1 DO oplot, time_list[i], obs_list[i], color = 0
  ENDIF

  ; ---------------- fit each curve ----------------
  fits_list = list() & trend_list = list() & flares_list = list()
  cred_up_list = list() & cred_down_list = list()
  params_list = list() & param_ci_list = list()
  maxpos_list = list() & n_ref_list = list()

  FOR i = 0, n_curves - 1 DO BEGIN

    t_abs = time_list[i]
    tt = t_abs - t_abs[0]                        ; time from start of the curve [min]
    yy = tbd_destep(t_abs, obs_list[i])
    n = n_elements(tt)
    col = strtrim(column_list[i], 2)

    ; noise level: st. dev. of residuals from Savitzky-Golay smoothing
    width = (n/5 LT 4) ? 2 : n/5
    sigma_noise = stddev(yy - convol(yy, savgol(width, width, 0, 4), /edge_truncate))

    ; reference peaks within the curve, plus a dummy peak if fewer than 3
    ref_pos = tbd_match_peaks(t_abs, t_peaks_solar)
    n_ref = ref_pos[0] EQ -1 ? 0 : n_elements(ref_pos)
    IF n_ref LT 3 THEN BEGIN
      dummy = tbd_dummy_position(ref_pos, n, rseed)
      maxpos = n_ref EQ 0 ? [dummy] : [ref_pos, dummy]
    ENDIF ELSE maxpos = ref_pos
    n_flares = n_elements(maxpos)
    print, 'Curve ' + strtrim(i + 1, 2) + '/' + strtrim(n_curves, 2) + $
      ' (column ' + col + '): ' + strtrim(n_ref, 2) + ' reference peak(s), ' + $
      strtrim(n_flares, 2) + ' peak(s) fitted'

    ; ---- initial guess
    n_par = 13 + 4*n_flares
    params = dblarr(n_par)
    a1 = 0.5d*(max(yy) - min(yy))
    a2 = 0.1d*a1
    p1 = 4d*max(tt)                              ; ~quarter of the 1st period observed
    params[0:12] = [mean(yy), a1, p1, 0d, a2, p1/2, 0d, 0.5*a2, p1/3, 0d, 0.25*a2, p1/4, 0d]
    params[[2, 5, 8, 11]] = (params[[2, 5, 8, 11]] > per_range[0]) < per_range[1]
    FOR k = 0, n_flares - 1 DO $
      params[13 + 4*k : 16 + 4*k] = [0.1d*sigma_noise, tt[maxpos[k]], 1d, 10d]

    ; ---- priors (uniform)
    priors = objarr(n_par)
    priors[0] = prior_uniform(c0_range[0], c0_range[1])
    FOR j = 0, 3 DO BEGIN
      priors[1 + 3*j] = prior_uniform(amp_range[0], amp_range[1])
      priors[2 + 3*j] = prior_uniform(per_range[0], per_range[1])
      priors[3 + 3*j] = prior_uniform(-!dpi, !dpi)
    ENDFOR
    FOR k = 0, n_flares - 1 DO BEGIN
      b = 13 + 4*k
      tk = tt[maxpos[k]]
      priors[b]     = prior_uniform(0d, fl_amp*sigma_noise)
      priors[b + 1] = prior_uniform(tk + fl_tpeak[0], tk + fl_tpeak[1])
      priors[b + 2] = prior_uniform(fl_rise[0], fl_rise[1])
      priors[b + 3] = prior_uniform(fl_decay[0], fl_decay[1])
    ENDFOR

    IF do_plot THEN BEGIN
      tbd_window, 1
      ymin = min(yy, max = ymax)
      plot, t_abs, yy, /xstyle, /ystyle, yrange = [ymin - 0.1*abs(ymin), ymax + 0.1*abs(ymax)], $
        color = 0, background = 255, thick = 2, $
        xtitle = 'Time [UT min]', ytitle = 'TEC [TECU]', $
        title = filename + ', column ' + col + ': initial guess'
      oplot, t_abs, trend_harmonic_flares(tt, params), color = 150
      IF n_ref GT 0 THEN oplot, t_abs[ref_pos], yy[ref_pos], psym = 4, symsize = 3, thick = 2, color = 60
      IF n_ref LT 3 THEN oplot, [t_abs[maxpos[-1]]], [yy[maxpos[-1]]], psym = 4, symsize = 3, thick = 2, color = 250
      write_png, dir + filename + '_' + col + '_init.png', tvrd(true = 1)
    ENDIF

    ; ---- MCMC fit; params returns the MAP estimate with the noise sigma appended
    fit = mcmc_fit(tt, yy, params, 'trend_harmonic_flares', priors = priors, $
      burn_in = burn_in, n_samples = n_samples, samples = samples, $
      confidence_level = conf_level, credible_intervals = param_ci)

    ; ---- credible band of the posterior predictive distribution, from a
    ;      random subset of posterior samples (model + Gaussian noise)
    ppd_seed = 100L
    ind = long(randomu(ppd_seed, n_ppd)*(n_elements(samples[0, *]) - 1))
    ppd = fltarr(n, n_ppd)
    FOR s = 0L, n_ppd - 1 DO BEGIN
      ps = samples[*, ind[s]]
      ppd[*, s] = trend_harmonic_flares(tt, ps) + ps[-1]*randomn(ppd_seed, n)
    ENDFOR
    dc = 0.5d*(1d - conf_level)
    up = dblarr(n) & down = dblarr(n)
    FOR j = 0L, n - 1 DO BEGIN
      q = cgpercentiles(ppd[j, *], percentiles = [dc, 1d - dc])
      down[j] = q[0] & up[j] = q[1]
    ENDFOR

    ; ---- background and peaks
    p_flares = params
    p_flares[[0, 1, 4, 7, 10]] = 0d              ; zero offset and harmonic amplitudes
    flares = trend_harmonic_flares(tt, p_flares)
    trend = fit - flares

    ; keep the dummy slot in the layout: NaN if no dummy was used
    IF n_ref GE 3 THEN BEGIN
      params = [params[0:-2], replicate(!values.d_nan, 4), params[-1]]
      param_ci = [param_ci[0:-2, *], replicate(!values.d_nan, 4, 2), param_ci[-1, *]]
    ENDIF

    fits_list.add, fit
    trend_list.add, trend
    flares_list.add, flares
    cred_up_list.add, up
    cred_down_list.add, down
    params_list.add, params
    param_ci_list.add, param_ci
    maxpos_list.add, maxpos
    n_ref_list.add, n_ref

    IF do_plot THEN BEGIN
      tbd_window, 2
      plot, t_abs, yy, /nodata, /xstyle, /ystyle, yrange = [min([down, yy]), max([up, yy])], $
        color = 0, background = 255, $
        xtitle = 'Time [UT min]', ytitle = 'TEC [TECU]', $
        title = filename + ', column ' + col + ': fit'
      oplot, t_abs, up, color = 60
      oplot, t_abs, down, color = 60
      oplot, t_abs, yy, color = 0
      oplot, t_abs, fit, color = 250
      oplot, t_abs, trend, color = 150
      oplot, t_abs[maxpos], fit[maxpos], psym = 4, symsize = 3, thick = 2, color = 60
      IF n_ref LT 3 THEN oplot, [t_abs[maxpos[-1]]], [fit[maxpos[-1]]], psym = 4, symsize = 3, thick = 2, color = 250
      write_png, dir + filename + '_' + col + '_fit.png', tvrd(true = 1)
    ENDIF

  ENDFOR

  ; ---------------- summary plots ----------------
  IF do_plot THEN BEGIN
    tbd_window, 3
    plot, [0], [0], /nodata, xrange = [time_min, time_max], yrange = yrange_all, $
      /xstyle, /ystyle, color = 0, background = 255, $
      xtitle = 'Time [UT min]', ytitle = 'TEC [TECU]', title = filename + ': data and fits'
    FOR i = 0, n_curves - 1 DO BEGIN
      oplot, time_list[i], obs_list[i], color = 0, thick = 3
      oplot, time_list[i], fits_list[i], color = 250
    ENDFOR
    write_png, dir + filename + '_all.png', tvrd(true = 1)

    tbd_window, 4, xsize = 1900, ysize = 800
    plot, [0], [0], /nodata, xrange = [time_min, time_max], yrange = yrange_dt, $
      /xstyle, /ystyle, color = 0, background = 255, xmargin = [7, 2], ymargin = [5, 2], $
      xtitle = 'Time [UT min]', ytitle = 'Delta TEC [TECU]', title = filename + ': detrended'
    colors = floor((lindgen(n_curves) + 1)*(254d/n_curves))
    FOR i = 0, n_curves - 1 DO BEGIN
      oplot, time_list[i], obs_list[i] - trend_list[i], color = 0
      oplot, time_list[i], flares_list[i], color = colors[i], thick = 2
    ENDFOR
    FOREACH tp, t_peaks_solar DO oplot, [tp, tp], yrange_dt, color = 250, thick = 3
    write_png, dir + filename + '_all_detrended.png', tvrd(true = 1)

    device, decomposed = decomposed0
  ENDIF

  ; ---------------- save ----------------
  sav_file = dir + filename + '_bayes.sav'
  save, time_list, obs_list, fits_list, trend_list, flares_list, $
    cred_up_list, cred_down_list, params_list, param_ci_list, $
    maxpos_list, n_ref_list, column_list, t_peaks_solar, dir, filename, $
    filename = sav_file
  print, 'Results saved to ' + sav_file

END

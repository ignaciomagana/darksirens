INVALID (2026-08-10): these scans consumed
gwsamples_bbh_whitelist_all_events.h5, which is filtered from the RAW
mixedcbc store -- p_pe lacks the gwcat processing (mass jacobian /
distance prior / source-frame conventions) the 44-event product carries.
Symptoms: complete=NaN, per_pixel railed at 97, sel railed at the 139
grid edge, |logL| ~ 1e6. Awaiting the owner's choice of the correct
259-event PE product + injection pairing before rerunning.

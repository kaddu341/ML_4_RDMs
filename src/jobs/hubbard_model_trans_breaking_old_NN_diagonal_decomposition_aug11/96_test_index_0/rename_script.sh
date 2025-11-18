for f in tMoTe2_2BPV_HF_model_*.pth; do
    mv -- "$f" "${f/tMoTe2/hubbard_model}"
done
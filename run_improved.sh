set -e
cd /home/yjlab/SCLC_3_multimodal_test
# wait for the lingering regime_test to free the GPU
while kill -0 1672168 2>/dev/null; do sleep 5; done
echo "=== REGIME_DONE $(date); memory smoke (all, bs=32, 1 fold, 2 ep) ==="
if python -c "
from train import TrimodalEvaluator
from ablation import make_factory, CONFIGS
TrimodalEvaluator(target='os',epochs=2,batch_size=32,max_folds=1,save_dir='outputs/mem_smoke_bs32',model_factory=make_factory(CONFIGS['all'])).run()
print('MEMSMOKE_OK')
" > mem_smoke_bs32.log 2>&1; then
  echo "=== MEM_OK; launching improved-regime ablation (bs=32, ep=60) ==="
  python ablation.py --target os  --batch_size 32 --epochs 60 --tag _improved
  python ablation.py --target pfs --batch_size 32 --epochs 60 --tag _improved --configs all,clin_report,clin_image
  echo "=== ALL_IMPROVED_DONE $(date) ==="
else
  echo "=== MEM_SMOKE_FAILED (likely OOM at bs=32); see mem_smoke_bs32.log ==="
  tail -3 mem_smoke_bs32.log
fi

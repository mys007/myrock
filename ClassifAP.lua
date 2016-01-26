require 'nn'

local ClassifAP = torch.class('ClassifAP')

function ClassifAP:__init()
	self.correct = {}
	self.scores = {}
   -- buffers
   self._target = torch.FloatTensor()
   self._prediction = torch.FloatTensor()	
   self._max = torch.FloatTensor()
   self._pred_idx = torch.LongTensor()
   self._targ_idx = torch.LongTensor()   
end


local function VOCap(rec,prec)
	--borrowed from https://github.com/fmassa/object-detection.torch/blob/fast-rcnn/utils.lua
  local mrec = rec:totable()
  local mpre = prec:totable()
  table.insert(mrec,1,0); table.insert(mrec,1)
  table.insert(mpre,1,0); table.insert(mpre,0)
  for i=#mpre-1,1,-1 do
      mpre[i]=math.max(mpre[i],mpre[i+1])
  end
  
  local ap = 0
  for i=1,#mpre-1 do
    if mrec[i] ~= mrec[i+1] then
      ap = ap + (mrec[i+1]-mrec[i])*mpre[i+1]
    end
  end
  return ap
end


function ClassifAP:compute(gtPosCount)
	--adapted from https://github.com/fmassa/object-detection.torch/blob/fast-rcnn/utils.lua
  if #self.scores==0 then return 0, torch.Tensor(), torch.Tensor() end
  
  local energy = torch.Tensor(self.scores)
  local correct = torch.Tensor(self.correct)
  
  local threshold,index = energy:sort(true)

  correct = correct:index(1,index)

  local n = threshold:numel()
  
  local recall = torch.zeros(n)
  local precision = torch.zeros(n)

  local num_correct = 0
  local count = gtPosCount or n

  for i = 1,n do
      --compute precision
      num_correct = num_correct + correct[i]
      precision[i] = num_correct / i;
      
      --compute recall
      recall[i] = num_correct / count
  end

  local ap = VOCap(recall, precision)

  return ap, recall, precision
end

function ClassifAP:add(correct, score)
   table.insert(self.correct, correct and 1 or 0)
   table.insert(self.scores, score)
end

function ClassifAP:batchAdd(predictions, targets, scores)
	-- borrowed from https://github.com/torch/optim/blob/master/ConfusionMatrix.lua
   local preds, targs, __
   self._prediction:resize(predictions:size()):copy(predictions)
   if predictions:dim() == 1 then
      -- predictions is a vector of classes
      preds = self._prediction
   elseif predictions:dim() == 2 then
      -- prediction is a matrix of class likelihoods
      if predictions:size(2) == 1 then
         -- or prediction just needs flattening
         preds = self._prediction:select(2,1)
      else
         self._max:max(self._pred_idx, self._prediction, 2)
         preds = self._pred_idx:select(2,1)
      end
   else
      error("predictions has invalid number of dimensions")
   end
      
   self._target:resize(targets:size()):copy(targets)
   if targets:dim() == 1 then
      -- targets is a vector of classes
      targs = self._target
   elseif targets:dim() == 2 then
      -- targets is a matrix of one-hot rows
      if targets:size(2) == 1 then
         -- or targets just needs flattening
         targs = self._target:select(2,1)
      else
         self._max:max(self._targ_idx, self._target, 2)
         targs = self._targ_idx:select(2,1)
      end
   else
      error("targets has invalid number of dimensions")
   end
   
   scores = scores:squeeze()
      
   --loop over each pair of indices
   for i = 1,preds:size(1) do
      self:add(preds[i]==targs[i], scores[i])
   end
end

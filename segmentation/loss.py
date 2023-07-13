import torch.nn as nn
import torch
import torch.nn.functional as F

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


# class DiceLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.epsilon = 1e-5

#     def forward(self, output, target):
#         assert output.size() == target.size(), "'input' and 'target' must have the same shape"
#         output = F.softmax(output, dim=1)
#         output = flatten(output)
#         target = flatten(target)
#         # intersect = (output * target).sum(-1).sum() + self.epsilon
#         # denominator = ((output + target).sum(-1)).sum() + self.epsilon

#         intersect = (output * target).sum(-1)
#         denominator = (output + target).sum(-1)
#         dice = intersect / denominator
#         dice = torch.mean(dice)
#         return 1 - dice
#         # return 1 - 2. * intersect / denominator

class DiceLoss(nn.Module):
	def __init__(self):
		super(DiceLoss, self).__init__()
 
	def	forward(self, input, target):
		N = target.size(0)
		smooth = 1
 
		input_flat = input.view(N, -1)
		target_flat = target.view(N, -1)
 
		intersection = input_flat * target_flat
 
		loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
		loss = 1 - loss.sum() / N
 
		return loss
 
class MulticlassDiceLoss(nn.Module):
	def __init__(self, num_classes, weights=None):
		super(MulticlassDiceLoss, self).__init__()

		self.dice_function = DiceLoss()
		self.num_classes = num_classes
		self.weights = torch.ones(self.num_classes)
		if weights is not None:
			self.weights = weights
			
 
	def forward(self, input, target):
 
		dice = DiceLoss()
		totalLoss = 0

		for i, w in enumerate(self.weights):
			totalLoss += dice(input[:,i], target[:,i]) * w

		return totalLoss
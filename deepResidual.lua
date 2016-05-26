require 'image'
require 'nn'
-- load image
image = image.load("image.jpg",3,"double")
image = image:reshape(1, 3, 400, 400)
print(image:size())

-- build model

local function basicblock(nInpputPlane, nOutputPlane)
   n = nn:Sequential()
   n:add(nn.SpatialConvolution(nInpputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1))
   n:add(nn.SpatialBatchNormalization(nOutputPlane))
   n:add(nn.ReLU(true))
   n:add(nn.SpatialConvolution(nOutputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1))
   n:add(nn.SpatialBatchNormalization(64))

   shortcut = nn:Sequential()
   shortcut:add(nn.SpatialConvolution(nInpputPlane, nOutputPlane, 1, 1, 1, 1))
   shortcut:add(nn.SpatialBatchNormalization(64))

   b = nn:Sequential()
   b:add(nn.ConcatTable()
            :add(n)
            :add(shortcut)
   )
   b:add(nn.CAddTable(ture))
   b:add(nn.ReLU(true))

   return b
end

local b = basicblock()
model = nn:Sequential()
model:add(nn.SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3))-- pad 3 keepp the image 400*400
model:add(nn.SpatialBatchNormalization(64))
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1)) -- 1*64*100*100
model:add(basicblock())
model:add(basicblock())
model:add(basicblock())
model:add(basicblock())
model:add(basicblock())
model:add(basicblock())

local outp = model:forward(image)
print(outp:size())

--[[
i = nn:Sequential()
c = nn:ConcatTable()
c:add(nn.Identity())
c:add(nn.Identity())

i:add(c)
i:add(nn.CAddTable(true))
]]
--block:add(nn.CAddTable(true))
--block:add(ReLU(true))

--[[

block:add(c)
block:add(nn:CAddTable(true))
block:add(ReLU(true))

]]
--[[
local outp = s:forward(image)
--print(outp:size())

local ioutp = i:forward(image)

local matrix1 = torch.Tensor(5,5):fill(2)
local outp2 = i:forward(matrix1)
print(outp2)

print(i)
print(ioutp:size())

cadd = nn.CSubTable(true)
test = {}
table.insert(test, torch.Tensor(3,3):fill(2))
table.insert(test, torch.Tensor(3,3):fill(3))
table.insert(test, torch.Tensor(3,3):fill(2))
local cadd_test = cadd:forward(test)
print(cadd_test)
]]

--[[
cg=nn.Concat(3)
cg:add(nn.Identity())
cg:add(nn.MulConstant(0))
]]
--[[
g = nn:Sequential()
g:add(nn.SpatialAveragePooling(1,1,2,2))
g:add(nn.Concat(2)
         :add(nn.Identity())
         :add(nn.MulConstant(0)))

local matrix2 = torch.Tensor(3,5,5):fill(3)

local outp3 = g:forward(matrix2)
print(outp3)
]]

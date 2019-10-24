test_loader_manual=torch.utils.data.DataLoader(dataset=test_dataset,shuffle=False)
#-----------debug img----------------
#-------------extract from dataloader to list-----------------
img_pack=[]
iter_n=0
for i, (image, target) in enumerate(test_loader_manual):
    # fig = plt.figure(0)
    # print("print(type(image)) = ",type(image))
    # print("print(image.size()) = ", image.size()) # must be  torch.Size([1, 28, 28])
    # print("print(type(image[0])) = ",type(image[0]))
    # print("print(image[0].size()) = ", image[0].size()) # must be  torch.Size([1, 28, 28])
    # print("print(type(target)) = ", type(target))
    # print("print(target.item()) = ",target.item())
    # img_pack.append((image,target))
    img_pack.append((image[0].numpy()[0],str(target[0].item())))
    # print(img_pack[i][1])
    # fig.canvas.set_window_title(img_pack[i][1])
    # fig=plt.imshow(img_pack[i][0],cmap='gray')
    # fig.canvas.set_window_title(str(target[0]))
    # fig = plt.imshow(image[0].numpy()[0], cmap='gray')
    # plt.show()
    iter_n+=1
print("iter_n = ",iter_n)
print("hola")

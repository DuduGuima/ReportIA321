
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_harmonics.examples import ShallowWaterSolver
import torch_harmonics as harmonics
from torch.profiler import profile, ProfilerActivity

from tqdm import tqdm

import matplotlib.pyplot as plt



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def l2loss_sphere(solver, prd, tar, relative=True, squared=False):
    """
    Loss function is already implemented
    """
    loss = solver.integrate_grid((prd - tar)**2, dimensionless=True).sum(dim=-1)
    if relative:
        loss = loss / solver.integrate_grid(tar**2, dimensionless=True).sum(dim=-1)
    
    if not squared:
        loss = torch.sqrt(loss)
    loss = loss.mean()

    return loss


def train(model_holder, 
          train_loader, 
          valid_loader, 
          optimizer,
          scheduler, 
          device, 
          num_epochs,
          current_epoch,
          save_path,
          check_weights=True):
    print('Device being used: ',torch.cuda.get_device_name(device))
    print('Memory before model to device ',torch.cuda.memory_allocated()/(1024**2))
    model_holder.model.to(device)  # Move model to device (CPU or GPU)
    solver = ShallowWaterSolver(model_holder.model.img_size[0],model_holder.model.img_size[1],dt=0, grid = model_holder.model.grid).to(device)
    print('Memory before epoch start ',torch.cuda.memory_allocated()/(1024**2))

    print("Initial weights + params:")
    model_holder.show_weights()
    
    for epoch in range(current_epoch,num_epochs):
        model_holder.model.train()  # Set model to training mode
        optimizer.zero_grad()
        epoch_loss = 0.0  # Track loss for the epoch
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move inputs and targets to the device
            #print('Data batch in')
            inputs, targets = inputs.to(device), targets.to(device)
           
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model_holder.model(inputs)
            # Compute loss
            loss = l2loss_sphere(solver,outputs, targets)
            #num_near_zeros = torch.sum(torch.abs(loss) < 1e-6).item()
            #print(f"Number of near-zero values in loss: {num_near_zeros}")
            # Backward pass
            loss.backward()
            # for param in model_holder.model.parameters():
            #     if param.grad is not None:
            # # Compute the L2 norm of the gradient
            #         grad_norm = param.grad.data.norm(2)
            # Clip grad
            torch.nn.utils.clip_grad_norm_(model_holder.model.parameters(), max_norm=1.0)
            # Update parameters
            optimizer.step()

            # Update lr
            scheduler.step()
            # Accumulate epoch loss
            epoch_loss += loss.item()

            
            #del loss
            del outputs

        if check_weights and (epoch%5==0):
                model_holder.show_weights()
        model_holder.model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valid_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                # Forward pass
                outputs = model_holder.model(inputs)
                # Compute loss
                loss = l2loss_sphere(solver,outputs, targets)

                valid_loss += loss.item()
        ave_val_loss = valid_loss/len(valid_loader)
        if ave_val_loss<model_holder.current_valid_ave:
            model_holder.set_ave_valid(ave_val_loss)
            model_holder.save_model(epoch+1,optimizer,scheduler,loss,save_path)
            print("Model saved!")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}, Val: {ave_val_loss:.4f}")
        



def evaluate_year(model_holder, test_loader,outcomes_holder,loss_holder, device):
    """
    Evaluates the model on the provided data.

    Args:
        model (torch.nn.Module): The trained PyTorch model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        criterion (torch.nn.Module, optional): Loss function to compute loss, if needed.
        device (str): Device to run the evaluation on ('cpu' or 'cuda').

    Returns:
        dict: A dictionary containing evaluation metrics such as loss and accuracy.
    """
    
    total_loss = 0.0
    solver = ShallowWaterSolver(model_holder.model.img_size[0],model_holder.model.img_size[1],dt=0, grid = model_holder.model.grid).to(device)

    model_holder.model.to(device)

    # tot_length = len(test_loader) + 6

    # output_length = tot_length//6



    # outcomes_holder = torch.empty((output_length,n_channels,lat,lon),dtype=torch.float32,device=device)
    # loss_holder = torch.empty(output_length,dtype=torch.float32,device=device)

    # outcomes_holder.to(device)
    # loss_holder.to(device)
    print('Output size = ',len(outcomes_holder))
    model_holder.model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation

        for idx,(inputs,targets) in enumerate(tqdm(test_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            output_index = idx//6
            if idx//6 >= len(outcomes_holder):
                break
            elif idx==0:
                # print('inputs ',inputs.is_cuda)
                outputs = model_holder.model(inputs)  # Forward pass
                outputs.to(device)
            elif idx%6==0:
                # print('whole outcomes ',outcomes_holder.is_cuda)
                # print('outcomes idx',outcomes_holder[idx-1].is_cuda)
                # print('shapes ',outcomes_holder[idx-1].shape)
                outputs = model_holder.model(outcomes_holder[output_index-1:output_index,:,:,:])
                outputs.to(device)
            else:
                continue
            outcomes_holder[output_index,:,:,:] = outputs.unsqueeze(0)

            loss = l2loss_sphere(solver,outputs,targets)

            loss_value = loss.item()
            
            loss_holder[output_index] = loss_value

    # Calculate average loss and accuracy
    
    return loss_holder,outcomes_holder


def evaluate_year_comp(model_holder, test_loader,outcomes_holder,gt_holder,loss_holder,index_holder, device):
    """
    Evaluates the model on the provided data, using another inference schema.
    Here, for each ground_truth point, 7 predictions are made, giving 48 hrs of predictions.

    Args:
        model (torch.nn.Module): The trained PyTorch model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        criterion (torch.nn.Module, optional): Loss function to compute loss, if needed.
        device (str): Device to run the evaluation on ('cpu' or 'cuda').

    Returns:
        dict: A dictionary containing evaluation metrics such as loss and accuracy.
    """
    
    total_loss = 0.0
    solver = ShallowWaterSolver(model_holder.model.img_size[0],model_holder.model.img_size[1],dt=0, grid = model_holder.model.grid).to(device)

    model_holder.model.to(device)

    print('Output size = ',len(outcomes_holder))
    model_holder.model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation

        for idx,(inputs,targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            output_index = idx//6
            if idx//6 >= len(outcomes_holder):
                break
            # elif (idx % 8)==0:
            #     # Uses ground_truth for points after 48h predictions
            #     outputs = model_holder.model(inputs)  # Forward pass
            #     outputs.to(device)
            elif idx%6==0:
                # In this case, we have a point in the 6 hours distance from the last one
                # print('whole outcomes ',outcomes_holder.is_cuda)
                # print('outcomes idx',outcomes_holder[idx-1].is_cuda)
                # print('shapes ',outcomes_holder[idx-1].shape)
                
                if (output_index%8) == 0 :
                    #print("truth point: ",idx,output_index)
                    outputs = inputs.clone()  # Forward pass
                    outputs.to(device)
                else:
                    outputs = model_holder.model(outcomes_holder[output_index-1:output_index,:,:,:])
                    outputs.to(device)
            else:
                continue
            index_holder[output_index] = idx
            outcomes_holder[output_index,:,:,:] = outputs.unsqueeze(0)
            gt_holder[output_index,:,:,:] = inputs.unsqueeze(0)
            loss = l2loss_sphere(solver,outputs,targets)

            loss_value = loss.item()
            
            loss_holder[output_index] = loss_value

    # Calculate average loss and accuracy
    
    return loss_holder,outcomes_holder,gt_holder,index_holder


def fine_tune(model_holder, 
          train_loader, 
          valid_loader, 
          optimizer,
          scheduler, 
          device, 
          num_epochs,
          current_epoch,
          save_path,
          tune_steps,
          dataset_step=1,
          check_weights=True):

    print('Device being used: ',torch.cuda.get_device_name(device))
    print('Memory before model to device ',torch.cuda.memory_allocated()/(1024**2))
    model_holder.model.to(device)  # Move model to device (CPU or GPU)
    solver = ShallowWaterSolver(model_holder.model.img_size[0],model_holder.model.img_size[1],dt=0, grid = model_holder.model.grid).to(device)
    print('Memory before epoch start ',torch.cuda.memory_allocated()/(1024**2))

    model_holder.set_best_val_loss(float('inf'))

    print("Initial weights + params:")
    model_holder.show_weights()
    
    for epoch in range(current_epoch,num_epochs):
        model_holder.model.train()  # Set model to training mode
        optimizer.zero_grad()
        epoch_loss = 0.0  # Track loss for the epoch
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move inputs and targets to the device
            # taraget is (B,C,tunesteps*nsteps,lat,lon)
            assert(len(targets.shape)>4)
            inputs, targets = inputs.to(device), targets.to(device)
           
            # Zero the parameter gradients
            optimizer.zero_grad()

            #first pass
            outputs = model_holder.model(inputs)
            loss = l2loss_sphere(solver,outputs, targets[:,:,0,:,:])
            # Forward pass until desired amplitude
            for i in range(1,tune_steps):
                outputs = model_holder.model(outputs.clone().detach())#detach here is just a check
                loss+=l2loss_sphere(solver,outputs,targets[:,:,dataset_step*i,:,:])
            # Compute loss
            #loss = l2loss_sphere(solver,outputs, targets)

            #num_near_zeros = torch.sum(torch.abs(loss) < 1e-6).item()
            #print(f"Number of near-zero values in loss: {num_near_zeros}")

            # Backward pass
            loss.backward()

            # for param in model_holder.model.parameters():
            #     if param.grad is not None:
            # # Compute the L2 norm of the gradient
            #         grad_norm = param.grad.data.norm(2)
            # Clip grad
            torch.nn.utils.clip_grad_norm_(model_holder.model.parameters(), max_norm=1.0)

            # Update parameters
            optimizer.step()

            # Update lr
            scheduler.step()

            # Accumulate epoch loss
            epoch_loss += loss.item()

            #del loss
            del outputs

        if check_weights and (epoch%5==0):
                model_holder.show_weights()
        model_holder.model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valid_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                # Forward pass
                outputs = model_holder.model(inputs)
                loss = l2loss_sphere(solver,outputs, targets[:,:,0,:,:])
                # Forward pass until desired amplitude
                for i in range(1,tune_steps):
                    outputs = model_holder.model(outputs.clone().detach())#detach here is just a check
                    loss+=l2loss_sphere(solver,outputs,targets[:,:,dataset_step*i,:,:])

                valid_loss += loss.item()
        ave_val_loss = valid_loss/(len(valid_loader)*tune_steps)
        if ave_val_loss<model_holder.current_valid_ave:
            model_holder.set_ave_valid(ave_val_loss)
            model_holder.save_model(epoch+1,optimizer,scheduler,loss,save_path)
            print("Model saved!")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / (len(train_loader)*tune_steps):.4f}, Val: {ave_val_loss:.4f}")



def mc_dropout(model_holder,
                       test_loader,
                       outcomes_holder,
                       n_instances,
                       index_holder,
                       device):
    """
    Evaluates the model on the provided data, using another inference schema.
    Here, for each ground_truth point, 7 predictions are made, giving 48 hrs of predictions.

    Args:
        model (torch.nn.Module): The trained PyTorch model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        criterion (torch.nn.Module, optional): Loss function to compute loss, if needed.
        device (str): Device to run the evaluation on ('cpu' or 'cuda').

    Returns:
        dict: A dictionary containing evaluation metrics such as loss and accuracy.
    """
    drop_layer = torch.nn.Dropout(p=0.1)
    total_loss = 0.0
    solver = ShallowWaterSolver(model_holder.model.img_size[0],model_holder.model.img_size[1],dt=0, grid = model_holder.model.grid).to(device)

    model_holder.model.to(device)
    instances=[]
    print('Output size = ',len(outcomes_holder))
    model_holder.model.train()  # Set the model to train mode to have dropouts
    with torch.no_grad():  # Disable gradient computation
        for i in range(n_instances):
            for idx,(inputs,targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                output_index = idx//6
                if idx//6 >= len(outcomes_holder):
                    break
                # elif (idx % 8)==0:
                #     # Uses ground_truth for points after 48h predictions
                #     outputs = model_holder.model(inputs)  # Forward pass
                #     outputs.to(device)
                elif idx%6==0:
                    # In this case, we have a point in the 6 hours distance from the last one
                    # print('whole outcomes ',outcomes_holder.is_cuda)
                    # print('outcomes idx',outcomes_holder[idx-1].is_cuda)
                    # print('shapes ',outcomes_holder[idx-1].shape)
                    
                    if (output_index%8) == 0 :
                        #print("truth point: ",idx,output_index)
                        outputs = inputs.clone()  # Forward pass
                        outputs.to(device)
                    else:
                        outputs = model_holder.model(outcomes_holder[output_index-1:output_index,:,:,:])
                        outputs.to(device)
                else:
                    continue
                index_holder[output_index] = idx
                outcomes_holder[output_index,:,:,:] = outputs.unsqueeze(0)
            instances.append(drop_layer(outcomes_holder))
        instance_mean = torch.stack(instances).mean(dim=0)
        instance_std = torch.stack(instances).std(dim=0)


    # Calculate average loss and accuracy
    
    return instance_mean,instance_std,index_holder
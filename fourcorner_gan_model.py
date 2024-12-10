import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class FourCornerGANModel(BaseModel):


    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """
        Add new dataset-specific options, and rewrite default values for existing options.
        添加新的数据集特定选项，并重写现有选项的默认值。

        Parameters:
            parser          -- original option parser 原始选项解析器
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options. 是否是训练阶段。你可以使用这个标志来添加训练特定或测试特定的选项。

        Returns:
            the modified parser. 修改后的解析器。
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout 默认CycleGAN不使用dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.1,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        return parser

    def __init__(self, opt):
        """
        Initialize the CycleGAN class.
        初始化CycleGAN类。

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions 存储所有实验标志；需要是BaseOptions的子类
        """
        BaseModel.__init__(self, opt)
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B',
                           'G_label_A', 'G_label_B', 'D_real_label_A', 'D_fake_label_A', 'D_real_label_B',
                           'D_fake_label_B']

        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')
        self.visual_names = visual_names_A + visual_names_B
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # 定义网络（包括生成器和判别器）
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:
                assert (opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionLabel = torch.nn.MSELoss()
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr_G, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr_D, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        从数据加载器中解包输入数据并执行必要的预处理步骤。

        Parameters:
            input (dict): include the data itself and its metadata information. 包括数据本身及其元数据信息。
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_A_label = input['A_label' if AtoB else 'B_label'].to(self.device)
        self.real_B_label = input['B_label' if AtoB else 'A_label'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """
        Run forward pass; called by both functions <optimize_parameters> and <test>.
        运行前向传递；由<optimize_parameters>和<test>函数调用。
        """
        noise_A = torch.randn(self.real_A.size()).to(self.device)
        noise_B = torch.randn(self.real_B.size()).to(self.device)

        self.fake_B = self.netG_A(self.real_A + noise_A)  # G_A(A + noise)
        self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A + noise))

        self.fake_A = self.netG_B(self.real_B + noise_B)  # G_B(B + noise)
        self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B + noise))


    def backward_D_basic(self, netD, real, fake, real_label):
        """
        Calculate GAN loss for the discriminator
        计算判别器的GAN损失

        Parameters:
            netD (network)      -- the discriminator D 判别器D
            real (tensor array) -- real images 真实图像
            fake (tensor array) -- images generated by a generator 生成器生成的图像

        Return the discriminator loss. 返回判别器损失。
        """
        parameter_label = 0.5
        pred_real, real_pred_label = netD(real)
        pred_fake, fake_pred_label = netD(fake.detach())

        real_pred_label = real_pred_label.view(real_pred_label.size(0), -1)[:, :50]
        fake_pred_label = fake_pred_label.view(fake_pred_label.size(0), -1)[:, :50]

        loss_D_real = self.criterionGAN(pred_real, True)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D_real_label = self.criterionLabel(real_pred_label, real_label)
        loss_D_fake_label = self.criterionLabel(fake_pred_label, real_label)

        loss_D = 0.5 * (loss_D_real + parameter_label * loss_D_real_label) + 0.5 * (
                    loss_D_fake + parameter_label * loss_D_fake_label)
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """
        Calculate GAN loss for discriminator D_A
        计算判别器D_A的GAN损失
        """
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B, self.real_B_label)

    def backward_D_B(self):
        """
        Calculate GAN loss for discriminator D_B
        计算判别器D_B的GAN损失
        """
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A, self.real_A_label)

    def backward_G(self):
        """
        Calculate the loss for generators G_A and G_B
        计算生成器G_A和G_B的损失
        """
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        if lambda_idt > 0:
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        pred_fake_B, fake_B_label = self.netD_A(self.fake_B)
        pred_fake_A, fake_A_label = self.netD_B(self.fake_A)
        self.loss_G_A = self.criterionGAN(pred_fake_B, True)
        self.loss_G_B = self.criterionGAN(pred_fake_A, True)

        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        self.loss_G_label_A = self.criterionLabel(fake_B_label.view(fake_B_label.size(0), -1)[:, :50],
                                                  self.real_B_label) * lambda_A
        self.loss_G_label_B = self.criterionLabel(fake_A_label.view(fake_A_label.size(0), -1)[:, :50],
                                                  self.real_A_label) * lambda_B



        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_G_label_A + self.loss_G_label_B
        self.loss_G.backward()


    def optimize_parameters(self):
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights


